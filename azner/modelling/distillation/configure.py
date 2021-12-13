import os
import argparse
import torch

from azner.modelling.distillation.dataprocessor.seqcls_tasks import MnliProcessor
from azner.modelling.distillation.dataprocessor.seqtag_tasks import NerProcessor

import logging

logger = logging.getLogger(__name__)

PROCESSORS = {
    "ner": NerProcessor,
    "mnli": MnliProcessor,
}


def argparse_get_args():
    """
    Legacy function for setting args. Migrating to hydra

    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type",
        default=None,
        type=str,
        help="Which task? : [seqtag | seqcls], Default : Auto (set by task_name)",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        type=str,
        help="Path to labels.txt for NER (Should contain a list of B, I, O labels; one label per a line.",
    )
    parser.add_argument("--teacher_model", default=None, type=str, help="The teacher model dir.")
    parser.add_argument(
        "--student_model", default=None, type=str, required=True, help="The student model dir."
    )
    parser.add_argument(
        "--tokenizer", default=None, type=str, help="The name of the task to train."
    )
    parser.add_argument(
        "--task_name", default=None, type=str, required=True, help="The name of the task to train."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Whether to run test on the test set. (Without training)",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--train_batch_size", default=32, type=int, help="Total batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", default=32, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", "--wd", default=1e-4, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=float,
        help="Total number of training epochs to perform. -1 for unlimited",
    )
    parser.add_argument(
        "--num_train_steps",
        default=None,
        type=float,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.01,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. Use warmup_steps to overried."
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=None,
        type=int,
        help="Warmup steps (override warmup_proportion). If None: warmup_proportion will be used."
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass. "
        "[accumulate_grad_batches] parameter for pytorch_lightning.trainer.trainer class",
    )

    # added arguments
    parser.add_argument("--aug_train", action="store_true")
    parser.add_argument("--eval_step", type=int, default=500)
    parser.add_argument("--pred_distill", action="store_true")
    parser.add_argument("--data_url", type=str, default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument("--save_every_n_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--early_stop", action="store_true", help="Whether to use early_stop")
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=None)

    args = parser.parse_args()

    return args


def args_sanity_check(args):
    if args.task_name.lower() != args.task_name:
        args.task_name = args.task_name.lower()

    assert args.task_name in PROCESSORS, "Task not found: %s" % args.task_name

    # Save frequency : save_every_n_epochs and save_every_n_steps should be exclusive
    if args.save_every_n_steps is not None:
        assert (
            args.save_every_n_epochs is None
        ), "One of args.save_every_n_steps or args.save_every_n_epochs should be None: %s, %s" % (
            args.save_every_n_steps,
            args.save_every_n_epochs,
        )
        assert float(
            args.save_every_n_steps
        ).is_integer(), "args.save_every_n_steps should be integer"
    elif args.save_every_n_epochs is not None:
        assert float(
            args.save_every_n_epochs
        ).is_integer(), "args.save_every_n_epochs should be integer"
    else:
        args.save_every_n_epochs = 1
    logger.info(
        "Save frequency: args.save_every_n_steps: {}, args.save_every_n_epochs: {}".format(
            args.save_every_n_steps, args.save_every_n_epochs
        )
    )


def prepare_output_dir(output_dir):
    """
    Check destination location and make a folder at output_dir.

    :param output_dir: Destination for the distilled model.
    :type output_dir: Path or str
    """
    if os.path.exists(output_dir) and os.listdir(output_dir):
        if len(os.listdir(output_dir)) > 1:
            raise ValueError(
                "Output directory ({}) already exists and is not empty.".format(output_dir)
            )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def set_accelerator(args):
    """
    Prepare devices. If args.gpu == 0, use CPU.

    """
    if args.gpus is not None and args.gpus > 0:
        devices_count = args.gpus
        device_type = "gpu"
    elif args.gpus == 0 or args.no_cuda:  # for debug -> CPU only
        devices_count = None  # CPU only
        device_type = None
    else:
        devices_count = "auto"  # Use GPUs if there are
        device_type = "auto"
    n_gpu = torch.cuda.device_count()
    logger.info(
        "device: {}, avaliable gpus: {}, # of gpus used: {}".format(
            device_type, n_gpu, devices_count
        )
    )
    return device_type, devices_count


def additional_task_params(args):
    """
    Preset for tasks. Returns output_mode and metric for the task.

    :param args:
    :param task_name:  Supports "ner" only at the moment.
    :type task_name: Str
    :return: output_mode and metric
    :rtype: tuple
    """
    output_modes = {
        "ner": "seqtag",
        "mnli": "classification",
    }
    # intermediate distillation default parameters
    default_params = {
        "ner": {"task_type": "seqtag", "num_train_epochs": 5},  # TODO
        "mnli": {"task_type": "seqcls", "num_train_epochs": 5},
    }
    metrics = {
        "ner": "entity_f1",
        "mnli": "accuracy"
        # TODO
    }

    if not args.pred_distill and not args.do_eval:
        if args.task_name in default_params:
            args.num_train_epoch = default_params[args.task_name]["num_train_epochs"]
            logger.info(
                "Overriding args.num_train_epoch as: {} (preset)".format(args.num_train_epoch)
            )

    if (args.task_type is None) and (args.task_name in default_params):
        args.task_type = default_params[args.task_name]["task_type"]

    if args.task_type not in ["seqtag", "seqcls"]:
        raise ValueError("Wrong task_type! Should be one of seqtag or seqcls")

    if args.metric is None or (args.metric not in ["entity_f1", "accuracy"]):
        args.metric = metrics[args.task_name]
        logger.info("Overriding args.metric as: {} (preset)".format(args.metric))

    return output_modes[args.task_name]


def set_metric_options(metric=None):
    """
    Set ckpt_filename, monitor_metric, monitor_mode for ModelCheckpoint and EarlyStopping classes.

    :param metric: One of "entity_f1", "accuracy". defaults to None ("accuracy")
    :type metric: Str, optional
    :return: ckpt_filename, monitor_metric, monitor_mode
    :rtype: tuple
    """
    if metric == "entity_f1":
        ckpt_filename = (
            "student_model-{epoch:02d}-{valF1:.4f}-{validation_loss_epoch:.3f}-{step:05d}"
        )
        monitor_metric = "valF1"
        monitor_mode = "max"
    else:
        ckpt_filename = "student_model-{epoch:02d}-{validation_loss_epoch:.3f}-{step:05d}"
        monitor_metric = "validation_loss_epoch"
        monitor_mode = "min"

    return ckpt_filename, monitor_metric, monitor_mode
