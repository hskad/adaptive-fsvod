# One-page report

`main.tex` is the submitted one-page report. It uses the instructor's template
(`template_original.tex`) without changes to its formatting: the preamble
(document class, margins, packages, line spacing, heading spacing, table and
caption settings) is identical to the template, list spacing uses the template
defaults, and the results table keeps the template's `@{}X c ...@{}` layout
with `\makecell` headers (columns: AP, AP50, AP75, Params, Latency). Only the
placeholder content was replaced.

Numbers beyond the main score study come from committed records:
latency from `results/score_calibration/latency.json`, the objectness/class-confidence
separation (Section III) from `results/score_calibration/objectness_separation.json`,
and the gain excluding bear from `results/score_calibration/per_class.csv`.

- Team No: 26
- Members: Daksh Agarwal (230101123), Kartik Maheshwari (230101116),
  Aditya Shukla (230101115), Suryadev Vinod (230121059), Sanket Prakash (230103121)
- PDF file name: `Deep_Learners_26_DLCV_Project_Report.pdf`

## Overleaf

Upload `Deep_Learners_26_DLCV_Project_Report_Overleaf.zip` as a new project,
open `main.tex`, and select XeLaTeX to match the local build. Recompile, confirm
exactly one A4 page, and download the PDF with the name above.
`template_original.tex` is a reference copy; compile `main.tex`.

## Local build

The PDF was compiled with Tectonic 0.17.0 (XeTeX):

```bash
tectonic --keep-logs main.tex
```

Rename `main.pdf` to the required name. The default `article` class ignores the
template's `8pt` option; this is the template's own behaviour and was left as is.
The build produces one A4 page with no overfull or underfull boxes.
`validation.json` records the source/template/PDF hashes, page count, summary
length, and metric checks.
