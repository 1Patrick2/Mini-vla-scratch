"""Tests for key_utils."""

import pytest

from mini_vla.datasets.key_utils import (
    find_first_key,
    get_by_key,
    has_key,
    list_available_keys,
)


class TestGetByKey:
    def test_flat_key(self):
        sample = {"observation.image": "val"}
        assert get_by_key(sample, "observation.image") == "val"

    def test_nested_key(self):
        sample = {"observation": {"image": "val"}}
        assert get_by_key(sample, "observation.image") == "val"

    def test_deep_nested(self):
        sample = {"a": {"b": {"c": "deep"}}}
        assert get_by_key(sample, "a.b.c") == "deep"

    def test_flat_preferred_over_nested(self):
        sample = {"observation.image": "flat", "observation": {"image": "nested"}}
        assert get_by_key(sample, "observation.image") == "flat"

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            get_by_key({"a": 1}, "observation.image")


class TestHasKey:
    def test_flat_exists(self):
        assert has_key({"observation.image": "v"}, "observation.image") is True

    def test_nested_exists(self):
        assert has_key({"obs": {"img": "v"}}, "obs.img") is True

    def test_missing(self):
        assert has_key({"a": 1}, "b.c") is False


class TestFindFirstKey:
    def test_first_match(self):
        sample = {"a": 1, "b": 2}
        assert find_first_key(sample, ["x", "a", "b"]) == "a"

    def test_none_match(self):
        assert find_first_key({"a": 1}, ["x", "y"]) is None

    def test_nested_candidate(self):
        sample = {"observation": {"image": "img", "state": "st"}}
        assert find_first_key(sample, ["obs.image", "observation.image"]) == "observation.image"


class TestListAvailableKeys:
    def test_flat_sample(self):
        keys = list_available_keys({"a": 1, "b": 2})
        assert "a" in keys
        assert "b" in keys

    def test_nested_flattened(self):
        sample = {"observation": {"image": "img", "state": "st"}}
        keys = list_available_keys(sample)
        assert "observation.image" in keys
        assert "observation.state" in keys

    def test_nested_not_flattened(self):
        sample = {"observation": {"image": "img"}}
        keys = list_available_keys(sample, flatten_nested=False)
        assert "observation" in keys
        assert "observation.image" not in keys
