import argparse
import shutil
import zipfile

import pytest

from adaptive_fsvod.data import checksum, read_json, write_json
from adaptive_fsvod.pilot import bundle, run_matrix, summarize


def test_runner_resume_collect_and_archive(corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    root = tmp_path / "experiment"
    (root / "pilot").mkdir(parents=True)
    for split in ("train", "val", "test"):
        for filename in (f"{split}.json", f"{split}_episodes.json"):
            shutil.copyfile(corpus / filename, root / "pilot" / filename)
    shutil.copytree(corpus / "cache", root / "cache")
    args = argparse.Namespace(experiment_dir=root, seeds=[42], config_dir=corpus,
                              steps=2, eval_every=1, device="cpu")
    run_matrix(args)
    checkpoint = root / "runs/fixed_seed42/last.pt"
    before = checksum(checkpoint)
    log = root / "runs/fixed_seed42/training.jsonl"
    before_log = log.read_bytes()
    run_matrix(args)
    assert checksum(checkpoint) == before
    assert log.read_bytes() == before_log
    result = summarize(root, args.seeds)
    assert len(result["rows"]) == 2
    assert result["aggregates"]["fixed"]["AP"]["sample_std"] is None
    assert result["protocol"]["synthetic"] is True
    assert result["paired_AP_differences"]["42"] == pytest.approx(
        result["rows"][1]["AP"] - result["rows"][0]["AP"])
    # Source images must stay out even if someone places them under pilot/.
    images = root / "pilot/images"
    images.mkdir()
    (images / "private.png").write_bytes(b"image")
    archive = bundle(root, args.seeds)
    assert checksum(archive["path"]) == archive["sha256"]
    with zipfile.ZipFile(archive["path"]) as saved:
        assert "runs/fixed_seed42/best.pt" in saved.namelist()
        assert "exports/pilot_summary.json" in saved.namelist()
        assert not any("/images/" in path or "/frames/" in path for path in saved.namelist())
    wrong = root / "runs/adaptive_seed42/test/metrics.json"
    metrics = read_json(wrong)
    metrics["episodes_sha256"] = "different"
    write_json(wrong, metrics)
    with pytest.raises(ValueError, match="episodes_sha256"):
        summarize(root, args.seeds)
