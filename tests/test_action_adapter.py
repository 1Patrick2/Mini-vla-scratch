import unittest

from mini_vla.robot_interface.action_adapter import clip_action


class ActionAdapterTest(unittest.TestCase):
    def test_clip_action_limits_each_dimension(self):
        self.assertEqual(clip_action([0.2, -2.0, 3.0], limit=1.0), [0.2, -1.0, 1.0])

    def test_clip_action_rejects_negative_limit(self):
        with self.assertRaisesRegex(ValueError, "limit must be non-negative"):
            clip_action([0.1], limit=-1.0)