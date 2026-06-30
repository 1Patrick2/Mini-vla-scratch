import unittest

from mini_vla.config import load_config


class ConfigLoaderTest(unittest.TestCase):
    def test_load_config_merges_base_and_override(self):
        config = load_config("configs/train/debug.yaml")

        self.assertEqual(config["model"]["name"], "mini_vla_cnn_gru")
        self.assertEqual(config["data"]["batch_size"], 4)
        self.assertEqual(config["train"]["epochs"], 1)