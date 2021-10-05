from hydra import initialize_config_module, compose

from azner.runner.azner_runner import AZNerRunner


def test_azner_runner():
    with initialize_config_module(config_module="azner.conf"):
        cfg = compose(config_name='config',overrides=[
            'ray=local'
        ])
        runner = AZNerRunner(cfg)
        assert runner.ner('hello') == "not implemented"
