"""Tests for dataset split manifest."""


import pytest

from mini_vla.datasets.splits import (
    EpisodeSplit,
    collect_episode_ids,
    create_episode_split,
    filter_samples_by_episode,
    load_split,
    save_split,
    validate_split,
)


def _make_samples(n=16):
    return [
        {"action": [float(i)], "episode_index": i // 4, "frame_index": i % 4}
        for i in range(n)
    ]


class TestCollectEpisodeIds:
    def test_collects_unique_sorted(self):
        samples = _make_samples(16)  # episodes 0,1,2,3
        ids = collect_episode_ids(samples)
        assert ids == [0, 1, 2, 3]

    def test_empty(self):
        assert collect_episode_ids([]) == []


class TestCreateEpisodeSplit:
    def test_no_overlap(self):
        samples = _make_samples(32)
        split = create_episode_split(samples, dataset_name="test", train_ratio=0.8)
        train_set = set(split.train_episode_ids)
        eval_set = set(split.eval_episode_ids)
        assert len(train_set & eval_set) == 0

    def test_reproducible_seed(self):
        samples = _make_samples(32)
        s1 = create_episode_split(samples, "test", seed=42)
        s2 = create_episode_split(samples, "test", seed=42)
        assert s1.train_episode_ids == s2.train_episode_ids
        assert s1.eval_episode_ids == s2.eval_episode_ids

    def test_different_seed_different(self):
        samples = _make_samples(32)
        s1 = create_episode_split(samples, "test", seed=42)
        s2 = create_episode_split(samples, "test", seed=99)
        # Very unlikely that different seeds produce the exact same split
        if len(s1.train_episode_ids) == len(s2.train_episode_ids):
            assert s1.train_episode_ids != s2.train_episode_ids


class TestFilterSamplesByEpisode:
    def test_filter_keeps_correct(self):
        samples = _make_samples(16)
        filtered = filter_samples_by_episode(samples, [0, 1])
        for s in filtered:
            assert s["episode_index"] in (0, 1)

    def test_filter_excludes_others(self):
        samples = _make_samples(16)
        filtered = filter_samples_by_episode(samples, [0])
        assert len(filtered) == 4
        for s in filtered:
            assert s["episode_index"] == 0


class TestSaveLoadSplit:
    def test_round_trip(self, tmp_path):
        split = EpisodeSplit(
            dataset_name="test",
            repo_id="lerobot/test",
            seed=42,
            train_episode_ids=[0, 1, 2],
            eval_episode_ids=[3],
        )
        path = save_split(split, tmp_path / "split.json")
        loaded = load_split(path)
        assert loaded.dataset_name == "test"
        assert loaded.train_episode_ids == [0, 1, 2]
        assert loaded.eval_episode_ids == [3]


class TestValidateSplit:
    def test_valid_passes(self):
        split = EpisodeSplit(dataset_name="t", repo_id="r", seed=0,
                             train_episode_ids=[0, 1], eval_episode_ids=[2])
        validate_split(split)  # should not raise

    def test_overlap_raises(self):
        split = EpisodeSplit(dataset_name="t", repo_id="r", seed=0,
                             train_episode_ids=[0, 1], eval_episode_ids=[0])
        with pytest.raises(ValueError, match="overlap"):
            validate_split(split)

    def test_empty_train_raises(self):
        split = EpisodeSplit(dataset_name="t", repo_id="r", seed=0,
                             train_episode_ids=[], eval_episode_ids=[0])
        with pytest.raises(ValueError, match="empty"):
            validate_split(split)

    def test_empty_eval_raises(self):
        split = EpisodeSplit(dataset_name="t", repo_id="r", seed=0,
                             train_episode_ids=[0], eval_episode_ids=[])
        with pytest.raises(ValueError, match="empty"):
            validate_split(split)
