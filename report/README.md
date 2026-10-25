# One-page report draft

The source in `main.tex` uses the instructor's section structure, fonts,
heading settings, and booktabs table style. The original template is retained
in `template_original.tex`. At the user's request, the top and bottom margins
are now 0.30 inch (originally 0.15 inch); side margins remain 0.55 inch.
This is an explicit margin change from the original template.

Local body spacing uses 1.5 pt between list items, no extra paragraph gap inside
items, 5 pt around full display equations (3 pt for short displays), normal
separation below the results table, and a small gap before the references.
Each reference starts on a separate line, with hanging indentation if it wraps.
Fonts and font sizes remain unchanged. The table keeps five balanced columns,
with AP/AP50/AP75 grouped under Detection performance and trainable head size
kept separate. The Method header and all entries are centered. The scientific text, equations, and result values are unchanged.
No latency improvement was measured or claimed.

The draft uses team name **Deep_Learners**. No team number was supplied. The
five member names and roll numbers were deferred by the user. Replace the
visible member placeholder before submission, then recompile and check the
page count again. The PDF file name is:

`Deep_Learners_26_DLCV_Project_Report.pdf`

## Overleaf

Upload the supplied `Deep_Learners_26_DLCV_Project_Report_Overleaf.zip` as a new
project, open `main.tex`, and select XeLaTeX to match the local build. Add the
five names and roll numbers in the heading. Recompile, confirm exactly one A4
page, and download the PDF with the required name. Preserve the font sizes
and check the one-page layout after editing. `template_original.tex` is a reference copy;
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
font-size repair was added. Margin and spacing refinements are listed above. `validation.json` records the
source/template/PDF hashes, page count, summary length, and metric checks.

The draft has a 71-word paper summary, the exact score rule and ranking
condition, the validation-only exponent selection, all three mean AP metrics,
seed variability, the paired video-bootstrap interval, and the reduced setup's
limits. It credits established objectness scoring and states that bear supplies
most of the mean gain. It does not claim full-paper benchmark reproduction.

The local PDF and Overleaf export are delivered under the workspace's
`output/pdf/centered_table/` directory. The report source is version-controlled here. The
original instructor ZIP in the workspace remains unchanged.
