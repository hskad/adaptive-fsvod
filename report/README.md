# One-page report draft

The source in `main.tex` fills the instructor's template. Its entire preamble,
including margins, font settings, spacing, packages, and table settings, is
byte-for-byte identical to `template_original.tex`. The enumerated section
structure and table style are retained. At the user's request, local body
spacing was refined: 3 pt between list items, no extra paragraph gap inside
items, 5 pt around full display equations (3 pt for short displays), normal
separation below the results table, and a small gap before the references.
Fonts, margins, section-heading settings, and table style are unchanged.
The scientific text, equations, and result values are unchanged. Table headings now report AP, AP50,
AP75, and trainable head size. No latency improvement was measured or claimed.

The draft uses team name **Deep_Learners**. No team number was supplied. The
five member names and roll numbers were deferred by the user. Replace the
visible member placeholder before submission, then recompile and check the
page count again. The PDF file name is:

`Deep_Learners_26_DLCV_Project_Report.pdf`

## Overleaf

Upload the supplied `Deep_Learners_26_DLCV_Project_Report_Overleaf.zip` as a new
project, open `main.tex`, and select XeLaTeX to match the local build. Add the
five names and roll numbers in the heading. Recompile, confirm exactly one A4
page, and download the PDF with the required name. Do not change template
margins, font sizes, or spacing. `template_original.tex` is a reference copy;
compile `main.tex`.

The separate Overleaf export contains only `main.tex` and this README, so it
has one document entry point. No bibliography download or external image is
required. All three reference links and the project URL are active in the PDF.

## Local build and checks

The reviewed PDF was compiled with Tectonic 0.17.0 (XeTeX):

```bash
tectonic --keep-logs main.tex
```

Rename `main.pdf` to the required name. The default `article` class ignores the
instructor's `8pt` option; this original template behavior was retained. No
font-size repair or layout override was added. `validation.json` records the
source/template/PDF hashes, page count, summary length, and metric checks.

The draft has a 71-word paper summary, the exact score rule and ranking
condition, the validation-only exponent selection, all three mean AP metrics,
seed variability, the paired video-bootstrap interval, and the reduced setup's
limits. It credits established objectness scoring and states that bear supplies
most of the mean gain. It does not claim full-paper benchmark reproduction.

The local PDF and Overleaf export are delivered under the workspace's
`output/pdf/revised/` directory. The report source is version-controlled here. The
original instructor ZIP in the workspace remains unchanged.
