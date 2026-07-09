"""Tests for shape_meta utility functions."""

from __future__ import annotations

import pytest

from mini_vla.datasets.shape_meta import (
    get_action_dim,
    get_action_horizon,
    get_obs_horizon,
    get_state_dim,
)


class TestGetActionDim:
    def test_from_shape_meta(self):
        config = {"shape_meta": {"action": {"dim": 14}}}
        assert get_action_dim(config) == 14

    def test_from_model_cfg(self):
        config = {"model": {"action_dim": 4}}
        assert get_action_dim(config) == 4

    def test_shape_meta_preferred_over_model(self):
        config = {"shape_meta": {"action": {"dim": 14}}, "model": {"action_dim": 2}}
        assert get_action_dim(config) == 14

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="action_dim"):
            get_action_dim({"model": {}})

    def test_empty_config_raises(self):
        with pytest.raises(ValueError, match="action_dim"):
            get_action_dim({})


class TestGetStateDim:
    def test_from_shape_meta(self):
        config = {"shape_meta": {"state": {"dim": 14}}}
        assert get_state_dim(config) == 14

    def test_from_model_cfg(self):
        config = {"model": {"state_dim": 6}}
        assert get_state_dim(config) == 6

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="state_dim"):
            get_state_dim({})


class TestGetObsHorizon:
    def test_from_shape_meta(self):
        config = {"shape_meta": {"obs_horizon": 2}}
        assert get_obs_horizon(config) == 2

    def test_default(self):
        assert get_obs_horizon({}) == 1


class TestGetActionHorizon:
    def test_from_shape_meta(self):
        config = {"shape_meta": {"action_horizon": 4}}
        assert get_action_horizon(config) == 4

    def test_default(self):
        assert get_action_horizon({}) == 1
