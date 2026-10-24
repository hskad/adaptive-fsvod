# Dataset preparation

## Selected first experiment

Use a custom FSYTV-40 subset, derived from YouTube-VIS 2019. This is a new subset protocol, not the paper's original benchmark protocol. The initial budget is 6 train classes, 3 validation classes, and 3 test classes, with about 3 videos per class and every fourth available frame. The exact counts are recorded in `subset.json`.

The code converted the public FSYTV annotation archive successfully: 2,235 videos, 3,679 tracks, and 40 category names. The original archive SHA-256 is `667ae65f3ab1a88dd12efabe266f764143279b8dcca3d9c0722ddf96e8613845`.

Dataset sources:

- [FewX dataset links](https://github.com/fanq15/FewX)
- [FSYTV-40 annotation folder](https://drive.google.com/drive/folders/1a1PpfAxeYL7AbxYViDDnx7ACFtRohVL5)
- [YouTube-VIS provider](https://youtube-vos.org/dataset/vis/)
- [Dataset terms](https://youtube-vos.org/dataset/term/)
- [Working RGB archive mirror](https://huggingface.co/datasets/JiaHuang01/Youtubevos_train), pinned revision `7855adf7e2e8e9386b578f58d490f25c1886368d`.

The legacy image Drive link returned HTTP 404. An alternative public YouTube-VOS training archive contains all 249 RGB frame paths selected by this FSYTV pilot. YouTube-VIS was derived from YouTube-VOS. We therefore keep the FSYTV box labels and subset protocol and obtain the corresponding RGB images from that archive. No labels or class splits need to change.

The archive is 8,957,125,671 bytes. The downloader uses HTTP byte ranges to fetch its ZIP directory and only selected frames. It refuses responses that do not honor the requested range. A range cache supports interruption and reuse. Each downloaded image is checked against the ZIP CRC, annotation dimensions, and image decoding. `download_manifest.json` records its SHA-256 and the immutable source revision. Sample label overlays were also inspected locally.

The provider releases the data for non-commercial research. Its annotations have a CC BY 4.0 license. See the original terms above. Dataset images are not distributed through this GitHub repository.

## Commands

Run from the repository root. Create `data/` before downloading.

```bash
mkdir -p data
python -m gdown 1-m5Qcuxj4h8rryqnELtgnvOtTtWJ_Vys -O data/fsytv-annotations.zip
python -m adaptive_fsvod.convert_fsytv --archive data/fsytv-annotations.zip --output data/fsytv-boxes.json
```

Expected images:

```text
data/ytvis/train/JPEGImages/
  0043f083b5/
    00000.jpg
    ...
  <other-video-hash>/
    ...
```

The converter streams annotations and discards masks. Original train and test category IDs are mapped by name. The subset maker splits those categories again with seed 42. Videos containing categories assigned to different splits are excluded. Remaining classes need at least two labeled videos. Train and test names in the original files therefore do not define our new protocol.

```bash
python -m adaptive_fsvod.prepare --annotation data/fsytv-boxes.json --images-root data/ytvis/train/JPEGImages --output data/pilot --class-counts 6 3 3 --max-videos-per-class 3 --frame-stride 4 --seed 42 --skip-image-check
python -m adaptive_fsvod.download_frames --manifests data/pilot/train.json data/pilot/val.json data/pilot/test.json --images-root data/ytvis/train/JPEGImages --range-cache cache/download_ranges
```

Preparation first writes the subset manifests before images exist. The next command retrieves and verifies every required frame. The downloader fails if a selected path is absent or its dimensions differ. `--skip-image-check` alone does not establish that the dataset is ready to train. Require `verified_frames` to equal `matched_frames` in the final download record.

The pilot has 6/3/3 classes, 18/9/9 videos, and 131/50/68 selected frames for train/validation/test. These are data selection counts, not experiment results. Repeat both commands to reproduce the data selection and downloads.

The mirror has been checked for this pilot. A larger or different selection must pass the same preflight check; the code does not assume that every possible FSYTV frame is available there. Repeating the downloader validates existing frames and reuses cached ranges.

Alternatively, `prepare` accepts labeled native YouTube-VIS track annotations. Set `--annotation` to the provider's JSON. Do not use an unlabeled validation or test JSON.

## Portability

Manifests save the resolved image root for the machine that prepared them. Re-run preparation when moving raw images to another machine, or pass `--images-root` to cache construction. The annotation content and selected video IDs must remain the same. A changed manifest invalidates its cache and episode checksum.
