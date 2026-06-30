import unittest

from mini_vla.config import load_config
from mini_vla.training.trainer import Trainer


class TrainerTest(unittest.TestCase):
    def test_trainer_describes_loaded_config(self):
        config = load_config("configs/train/debug.yaml")
        trainer = Trainer(config)

        self.assertEqual(
            trainer.describe(),
            "Trainer(model=mini_vla_cnn_gru, data=toy_2d, epochs=1)",
        )