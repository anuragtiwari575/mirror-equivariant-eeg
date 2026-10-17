[README.md](https://github.com/user-attachments/files/33604877/README.md)
# Exact hemispheric equivariance in motor-imagery EEG decoding

Code and results for the paper. No number in the paper is typed by hand: every
figure quoted in the text is a LaTeX macro written by `make_paper_assets.py`
from the per-fit result file, so re-running the analysis regenerates every table,
figure and quoted value.

The manuscript itself is not distributed here.

## What is here

All files sit at the top level, which is also how they sit in the working
folder: the modules import one another as siblings, so the layout here is the
layout they run in.

    *.py                  the analysis pipeline and the scripts that write
                          every table and figure (37 modules)
    per_subject.csv       per-subject accuracies behind the paper's tables
    summary.txt           the run summary printed when the assets were built
    MANIFEST.txt          what was written, and from which result file
    PRE_REGISTRATION_Cho2017.md
                          the hypotheses, decision rules and exclusion rules
                          fixed in advance of decoding the held-out corpus

## The per-fit result file

`step24_subject_specific.csv` is the file every number in the paper is
computed from: one row per subject, model, epoch window, protocol and seed, 6448
rows in all. `make_paper_assets.py` reads it and writes every table, figure and
macro from it, so it is the file to check a reported number against, and the
file to replace if you refit anything.

## Completeness of the code

There are 37 modules here: the entry points below plus every module they import,
transitively. The closure was computed and checked, so nothing it imports is
absent. The working folder these came from holds 122 Python files; the other 85
belong to earlier, unrelated work and are deliberately not here.

Entry points, with what each produces:

    step24_subject_specific.py    the per-subject fits -> step24_subject_specific.csv
    run_all2.py                   drives the above in resumable parts
    make_paper_assets.py          numbers.tex, the table files, the figure PDFs,
                                  per_subject.csv, summary.txt, MANIFEST.txt
    complexity.py                 Table 5
    dataset_table.py              Table 1
    make_method_figure.py         the two schematic figures
    step23_model_screening.py     the earlier screening stage, kept for provenance

Third-party imports across the whole closure: numpy, scipy, scikit-learn,
matplotlib, torch, mne, moabb, braindecode, pyriemann.

## Data

No recording is redistributed here. All five corpora are public and are obtained
through MOABB: BCI Competition IV 2a and 2b, the PhysioNet Motor
Movement/Imagery database, Cho et al., and Dreyer et al. The loaders in
`step24_subject_specific.py` fetch them.

## Rebuilding the results

    1.  python step24_subject_specific.py        # or run_all2.py, below
    2.  python make_paper_assets.py --datasets all
    3.  python complexity.py                     # Table 5
    4.  python dataset_table.py                  # Table 1
    5.  python make_method_figure.py             # the two schematic figures

Step 1 is the expensive one. `run_all2.py` drives it in parts so a long
queue can be resumed: `--list` shows the parts and their state, `--from N`
starts at part N. The full set of fits in this paper is 6448 and took days on one
GPU, so expect to run it in stages rather than in one call.

Step 2 writes `corrected_results/paper_assets/` containing `numbers.tex`, the
table files, the figure PDFs, `per_subject.csv`, `summary.txt` and
`MANIFEST.txt`. `numbers.tex` holds one macro per quoted value, so any number in
the paper can be traced to the fit that produced it by looking up its macro
there.

## Utilities

    check_mirror.py     verifies that each subject's montage admits a
                             complete involution; run before fitting anything
    status.py           per corpus, window and model: how many fits exist,
                             how many hold per-trial scores and mirrored scores
    show24.py           inspects the per-fit result file
    prune_scores.py     drops stale or duplicated rows from it; dry-run by
                             default, writes a backup before `--apply`
    reset_dataset.py    clears one corpus's rows so it can be refitted
    step23_model_screening.py
                             the earlier screening stage, kept for provenance

## Provenance

The code here is the state of the pipeline that produced the tables and figures of
the paper. `step23_model_screening.py` is an earlier stage, kept because the
screening it did shaped which models were carried forward; it is not needed to
reproduce anything in the paper.

## Citation

If you use this code, please cite the paper. A BibTeX entry will be added here
once it has a DOI.

## License

The code and result files here are released under the MIT License; see
`LICENSE`.
