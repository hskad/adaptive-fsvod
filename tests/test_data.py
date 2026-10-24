import copy

import pytest

from adaptive_fsvod.cache import CachedEpisodes
from adaptive_fsvod.data import VideoManifest, make_episodes, read_json, validate_episodes, validate_splits


def test_image_loading_and_split_separation(corpus):
    manifests = [VideoManifest(corpus / f"{split}.json") for split in ("train", "val", "test")]
    validate_splits(manifests)
    assert manifests[0].validate_images() == {"classes": 2, "videos": 4, "frames": 12}
    with pytest.raises(ValueError, match="Categories overlap"):
        validate_splits([manifests[0], manifests[0]])


def test_deterministic_episodes_and_full_video_evaluation(corpus):
    manifest = VideoManifest(corpus / "val.json")
    cached = CachedEpisodes(manifest, corpus / "cache/val")
    first = make_episodes(manifest, cached.pool, seed=9)
    second = make_episodes(manifest, cached.pool, seed=9)
    assert first == second
    validate_episodes(manifest, first)
    for episode in first["episodes"]:
        assert episode["frame_indices"] == [0, 1, 2]
        frames, support, targets, _ = cached.load(episode)
        assert len(frames) == 3 and len(support) == 2
        assert targets[-1]["labels"].numel() == 0


def test_support_query_leakage_is_rejected(corpus):
    manifest = VideoManifest(corpus / "train.json")
    episode_file = read_json(corpus / "train_episodes.json")
    broken = copy.deepcopy(episode_file)
    episode = broken["episodes"][0]
    episode["support"][episode["classes"][0]][0]["video_id"] = episode["video_id"]
    with pytest.raises(ValueError, match="leakage"):
        validate_episodes(manifest, broken)


def test_cache_mismatch_is_rejected(corpus):
    with pytest.raises(ValueError, match="does not match"):
        CachedEpisodes(VideoManifest(corpus / "train.json"), corpus / "cache/test")


def test_insufficient_support_fails(corpus):
    manifest = VideoManifest(corpus / "train.json")
    cached = CachedEpisodes(manifest, corpus / "cache/train")
    with pytest.raises(ValueError, match="No valid episodes"):
        make_episodes(manifest, cached.pool, k_shot=100)
