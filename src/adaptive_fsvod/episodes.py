"""Write fixed episode files before comparing methods."""

import argparse

from .cache import CachedEpisodes
from .data import VideoManifest, make_episodes, validate_episodes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--n-way", type=int, default=2)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--max-frames", type=int, help="Omit for full selected query videos")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = VideoManifest(args.manifest)
    cache = CachedEpisodes(manifest, args.cache)
    episodes = make_episodes(manifest, cache.pool, args.episodes, args.n_way, args.k_shot, args.max_frames, args.seed)
    validate_episodes(manifest, episodes)
    write_json(args.output, episodes)
    print(f"Saved {len(episodes['episodes'])} fixed episodes to {args.output}")


if __name__ == "__main__":
    main()
