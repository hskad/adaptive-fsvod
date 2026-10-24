# Dataset preparation

## Selected first experiment

Use a custom FSYTV-40 subset, derived from YouTube-VIS 2019. This is a new subset protocol, not the paper's original benchmark protocol. The initial budget is 6 train classes, 3 validation classes, and 3 test classes, with about 3 videos per class and every fourth available frame. The exact counts are recorded in `subset.json`.

The code converted the public FSYTV annotation archive successfully: 2,235 videos, 3,679 tracks, and 40 category names. The original archive SHA-256 is `667ae65f3ab1a88dd12efabe266f764143279b8dcca3d9c0722ddf96e8613845`.

Dataset sources:

- [FewX dataset links](https://github.com/fanq15/FewX)
- [FSYTV-40 annotation folder](https://drive.google.com/drive/folders/1a1PpfAxeYL7AbxYViDDnx7ACFtRohVL5)
- [YouTube-VIS provider](https://youtube-vos.org/dataset/vis/)
- [Dataset terms](https://youtube-vos.org/dataset/term/)

The legacy YouTube-VIS image Drive link returned HTTP 404 during setup on 24 October 2026. The annotation download worked. Images are not included in this repository. Obtain the YouTube-VIS 2019 training images from the provider's current distribution or a permitted mirror. Preserve the original video hash folders and file names. A mirror must contain the images, not only the annotation JSON. Record the mirror and version if one is used.

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
python -m adaptive_fsvod.prepare --annotation data/fsytv-boxes.json --images-root data/ytvis/train/JPEGImages --output data/pilot --class-counts 6 3 3 --max-videos-per-class 3 --frame-stride 4 --seed 42
```

The command checks each selected image and its dimensions. It fails if an image is missing. `--skip-image-check` is available only for an annotation-only planning check. It does not establish that the dataset is ready to train.

The checked annotation-only pilot has 6/3/3 classes, 18/9/9 videos, and 131/50/68 selected frames for train/validation/test. These are data selection counts, not experiment results. Recompute the counts with the command above.

Alternatively, `prepare` accepts labeled native YouTube-VIS track annotations. Set `--annotation` to the provider's JSON. Do not use an unlabeled validation or test JSON.

## Portability

Manifests save the resolved image root for the machine that prepared them. Re-run preparation when moving raw images to another machine, or pass `--images-root` to cache construction. The annotation content and selected video IDs must remain the same. A changed manifest invalidates its cache and episode checksum.
