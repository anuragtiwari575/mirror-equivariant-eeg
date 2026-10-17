# Hypotheses fixed before the fifth dataset is decoded

Written on 2026-10-12, before `step24_subject_specific.py` was run on Cho2017 (Cho et al. 2017, 52 subjects).
Cho2017 has not been touched by any analysis of this study: no model was trained on it, no number from it was
looked at, and no choice in the pipeline was made with it. The three statements below are therefore a
confirmation test, and every other analysis in the paper is exploratory and will be labelled as such.

The paper reports this file as it stands. If a statement is not confirmed, it is reported as not confirmed.

## What is fixed

**Data and protocol.** Cho2017, all 52 subjects, left hand against right hand, all EEG channels of the
recording, 4-38 Hz, 0 to 3 s after the cue, 250 Hz. Windows of 2 s every 0.2 s; a test trial is decided by the
mean output of its own windows. Protocol `session`: because Cho2017 holds one recording block per subject and
records the two classes in blocks, the test set is the last third of the trials of each class in recording
order, the training set is everything before it, and the per-channel mean and standard deviation are computed
on the training trials only. One seed (0), 200 epochs, Adam at 0.001, batches of 64 — the settings already
used for the other four datasets, unchanged.

**Statistics.** Paired over the 52 subjects, on the mean accuracy of each subject. Differences are reported in
accuracy points with a 95 % bootstrap interval (5000 resamples, seed 0) and a sign-flip permutation p-value,
the same functions as the rest of the study. Holm correction across the three hypotheses.

## H1 — the construction helps each backbone

For both backbones B in {EEGNet, ShallowConvNet}, the mean accuracy of `B + mirror` is higher than that of `B`.

Confirmed if, for both backbones, the paired difference is positive and the Holm-adjusted two-sided p-value is
below 0.05.

Prediction from the four datasets already analysed: between +2 and +10 points for EEGNet, between +4 and +11
for ShallowConvNet.

## H2 — the best model is ShallowConvNet with the construction

`shallow_mirror` has the highest mean accuracy of the eleven models on Cho2017.

Confirmed if it ranks first. Reported as not confirmed if any other model ranks above it, with that model named.

## H3 — how the symmetry is used matters little

For both backbones, the construction and channel reflection (the same symmetry as a training augmentation) are
within 2 accuracy points of each other.

Confirmed if, for both backbones, the 95 % interval of the paired difference `B + mirror` minus `B + CR` lies
entirely inside [-2, +2] points. Reported as inconclusive if the interval is wider than that band but contains
zero, and as not confirmed if the interval excludes [-2, +2] on either side.

With 52 subjects the interval may well be wider than the band; the inconclusive outcome is therefore a real
possibility and will be reported as such, not reinterpreted.

## What this file does not cover

The mirror-consistency measurement, the comparison of the two protocols, the per-subject analyses, the
components that were screened and rejected, and everything on the other four datasets are exploratory. They are
reported as exploratory wherever they appear.
