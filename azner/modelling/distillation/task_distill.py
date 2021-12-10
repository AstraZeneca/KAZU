# coding=utf-8
import sys
import csv
import logging

# Requires PL version at least 1.5.x (ModelCheckpoint)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import BertTokenizer

from azner.modelling.distillation.dataprocessor.dataprocessor import get_data_loader
from azner.modelling.distillation.configure import (
    argparse_get_args, 
    args_sanity_check, 
    prepare_output_dir,
    set_accelerator,
    additional_task_params,
    set_metric_options,
    PROCESSORS)

from azner.modelling.distillation.distillation.DistillModel import SequenceTaggingTaskSpecificDistillation
from azner.modelling.distillation.distillation.custom_checkpoint import CustomCheckpointIO


csv.field_size_limit(sys.maxsize)

def __get_logger():
    log_format = '%(asctime)s - %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format,
                        datefmt='%m/%d/%Y %H:%M:%S')
    fh = logging.FileHandler('train_run.log')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    __logger = logging.getLogger()

    return __logger
logger = __get_logger()

def main():
    args = argparse_get_args()
    args_sanity_check(args)
    prepare_output_dir(args.output_dir)
    device_type, devices_count = set_accelerator(args)
    output_mode = additional_task_params(args)
    logger.info('The args: {}'.format(args))

    pl.seed_everything(args.seed)

    processor = PROCESSORS[args.task_name]() # Pre-processing processors

    # Get labels
    if args.task_name == "ner":
        label_list = processor.get_labels(label_path=args.labels)
    else:
        label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    if args.do_test: # do_test only
        raise NotImplemented
    
    if not args.aug_train:
        train_dataloader, len_train_examples = get_data_loader(args, data_type="train", processor=processor, label_list=label_list, tokenizer=tokenizer, output_mode=output_mode)
    else: 
        train_dataloader, len_train_examples = get_data_loader(args, data_type="aug_train", processor=processor, label_list=label_list, tokenizer=tokenizer, output_mode=output_mode)
    eval_dataloader, _ = get_data_loader(args, data_type="development", processor=processor, label_list=label_list, tokenizer=tokenizer, output_mode=output_mode)

    # Scheduler config    
    if not args.pred_distill:
        schedule = 'none'
    elif devices_count == 'auto' or devices_count > 1:
        schedule = 'torchStepLR' # Multi-GPU : must use schedulers from torch
    else:
        schedule = 'warmup_linear'
    num_train_optimization_steps = int(len_train_examples / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    args.warmup_steps = int(args.warmup_proportion*num_train_optimization_steps) if args.warmup_steps==None else args.warmup_steps
    logger.info(
        "num_train_optimization_steps: {}, args.warmup_steps: {}, args.gradient_accumulation_steps: {}, args.warmup_proportion: {}".format(
        num_train_optimization_steps, args.warmup_steps, args.gradient_accumulation_steps, args.warmup_proportion)
    )

    if args.task_type == "seqtag":
        model = SequenceTaggingTaskSpecificDistillation(
            args=args, 
            label_list=label_list,
            schedule=schedule, 
            num_train_optimization_steps=num_train_optimization_steps,
            metric=args.metric
            )
    elif args.task_type == "seqcls":
        raise NotImplementedError
    else:
        raise ValueError("Wrong task_type: {}".format(args.task_type))

    # Adding callbacks
    callbacksList = []

    # checkpointing
    ckpt_filename, monitor_metric, monitor_mode = set_metric_options(args.metric)

    custom_checkpoint_io = CustomCheckpointIO(tokenizer=tokenizer)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=ckpt_filename,
        monitor=monitor_metric, mode=monitor_mode,
        save_top_k=100,
        save_last=True,
        every_n_train_steps=args.save_every_n_steps,
        every_n_epochs=args.save_every_n_epochs
    )
    callbacksList.append(checkpoint_callback)
    
    # Early stop
    if args.early_stop:
        earlystop_callback = EarlyStopping(
            monitor=monitor_metric, mode=monitor_mode,
            min_delta=0.00,
            patience=5 if args.save_every_n_steps != None else 3, # TODO
            verbose=True, 
            )
        callbacksList.append(earlystop_callback)

    # Train and evaluate
    trainer = pl.Trainer(
        plugins=[custom_checkpoint_io],
        callbacks=callbacksList,
        val_check_interval=args.save_every_n_steps if args.save_every_n_steps != None else 1.0,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        devices=devices_count, accelerator=device_type,
        max_epochs=args.num_train_epochs, max_steps=args.num_train_steps # whichever reaches earlier
        )
    trainer.fit(model, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()
