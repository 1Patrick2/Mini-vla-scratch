"""Tests for the config loader — merging, inline comments, etc."""

from __future__ import annotations

import tempfile
from pathlib import Path

from mini_vla.config import load_config
from mini_vla.config.loader import _strip_inline_comment


class TestInlineCommentStripping:
    def test_int_with_inline_comment(self):
        """Inline comment after int value should be stripped, value stays int."""
        body = (
            "model:\n"
            "  name: mini_vla\n"
            "  action_dim: 56  # flattened: 4 * 14\n"
            "train:\n"
            "  epochs: 1\n"
            "  lr: 0.001\n"
            "  device: cpu\n"
            "data:\n"
            "  dataset_type: toy_2d\n"
            "  data_root: data\n"
            "  batch_size: 2\n"
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        tmp.write(body)
        tmp.close()
        cfg = load_config(Path(tmp.name))
        assert cfg["model"]["action_dim"] == 56
        assert isinstance(cfg["model"]["action_dim"], int)

    def test_no_comment(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        tmp.write(
            "model:\n"
            "  name: mini_vla\n"
            "  action_dim: 56\n"
            "train:\n"
            "  epochs: 1\n"
            "  lr: 0.001\n"
            "  device: cpu\n"
            "data:\n"
            "  dataset_type: toy_2d\n"
            "  data_root: data\n"
            "  batch_size: 2\n"
        )
        tmp.close()
        cfg = load_config(Path(tmp.name))
        assert cfg["model"]["action_dim"] == 56

    def test_full_aloha_action_chunk_parses_correctly(self):
        """End-to-end: the ALOHA ActionChunk config must parse action_dim as int 56."""
        cfg = load_config(Path("configs/experiments/aloha_sim/action_chunk_bc_smoke.yaml"))
        assert cfg["model"]["action_dim"] == 56
        assert isinstance(cfg["model"]["action_dim"], int)


class TestStripInlineCommentUnit:
    def test_strips_simple(self):
        assert _strip_inline_comment("56  # comment") == "56"

    def test_keeps_quoted_hashtag_single(self):
        assert _strip_inline_comment("'move #1'  # task") == "'move #1'"

    def test_keeps_quoted_hashtag_double(self):
        assert _strip_inline_comment('"move #1"') == '"move #1"'

    def test_no_comment(self):
        assert _strip_inline_comment("56") == "56"

    def test_float_with_comment(self):
        assert _strip_inline_comment("0.001  # learning rate") == "0.001"

    def test_bool_with_comment(self):
        assert _strip_inline_comment("true  # freeze text encoder") == "true"
