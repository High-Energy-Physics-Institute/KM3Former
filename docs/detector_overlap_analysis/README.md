# Detector Overlap Analysis

## Summary

This analysis treats class `0/1/2` as physical `1/2/3` muons and focuses on why the true two-muon class collapses toward one- and three-muon predictions.

Main conclusion:

- the bottleneck is **mostly an information limit in detector response**, not a clean direction-based mistake
- true two-muon events occupy an overlap region in detector-summary space
- KM3Former improves over cheap detector-only baselines, but not by a large margin
- a Monte Carlo-only diagnostic check using hidden truth energies improves performance sharply, which implies that generator-level multiplicity information is only partially visible to the detector

Important distinction:

- all fair classifier comparisons in this report use detector-observable quantities only, such as hit activity, TOT-derived summaries, time span, geometry, and hit-direction summaries
- Monte Carlo truth energies are **not** available in a real experiment at inference time
- the MC-energy comparison is included only as a diagnostic upper-bound style sanity check to show that some class information exists in simulation truth but is not cleanly exposed by detector observables

## Artifacts

- [detector summary table](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/detector_summary_all_splits.csv)
- [detector feature overlap](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/detector_feature_overlap_test.csv)
- [baseline metrics](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/baseline_metrics.json)
- [model ambiguity summary](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/model_ambiguity_summary.json)
- [representative events](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/representative_events.csv)

Plots:

- [detector feature CDFs](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/plots/detector_feature_cdfs.png)
- [pairwise AUC heatmap](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/plots/pairwise_auc_heatmap.png)
- [baseline vs KM3Former confusion matrices](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/plots/confusion_comparison.png)
- [confidence by true class](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/plots/confidence_by_true_class.png)
- [class-1 reliability and margin](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/plots/class1_reliability_and_margin.png)
- [representative event panel](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/plots/representative_events_panel.png)

## Detector Overlap Clues

The strongest detector-summary features are the ones that track event activity and light yield, not direction. Top detector features by pairwise AUC strength on the test split:

| feature | ordered_sandwich | auc_strength_0_vs_1 | auc_strength_1_vs_2 | auc_strength_0_vs_2 |
| --- | --- | --- | --- | --- |
| active_hits | True | 0.623 | 0.686 | 0.788 |
| active_hit_fraction | True | 0.623 | 0.686 | 0.788 |
| radius_std | True | 0.623 | 0.685 | 0.787 |
| total_tot | True | 0.615 | 0.678 | 0.775 |
| max_tot_active | True | 0.541 | 0.604 | 0.644 |

Interpretation:

- if a feature is `ordered_sandwich = True`, the class medians are monotonic across `1 -> 2 -> 3` muons
- strong `1 vs 2` and `2 vs 3` AUC values mean the feature changes in the expected direction even before using a transformer
- the top-ranked features are primarily activity proxies such as hit count and TOT-derived summaries

## Baseline Comparison

Detector-only baselines on the test split:

- logistic regression: accuracy `0.484`, recall(2 muons) `0.243`
- random forest: accuracy `0.485`, recall(2 muons) `0.286`

Diagnostic upper-bound style check, not a fair detector-only baseline:

- logistic regression with detector features plus MC truth energies: accuracy `0.751`, recall(2 muons) `0.639`

What this means:

- this MC-augmented result is **not** a deployable model and should not be interpreted as a realistic experiment baseline
- it is only a sanity check: if hidden simulation truth improves separation a lot, then the difficult part of the task is that detector observables alone do not fully expose the same information

This gap is important:

- detector-only logistic regression is well below the transformer, so KM3Former is learning something useful
- but the detector-only baseline is still close enough to show that the overlap is already present in coarse hit-space
- the MC-augmented diagnostic rise indicates the missing signal is not purely a model-capacity issue
- the physically relevant verdict should therefore be based on the detector-only baselines and KM3Former, not on the MC-augmented diagnostic number

## Model Ambiguity

`exp_008_capacity_push`:

- true two-muon accuracy `0.290`
- median confidence for true two-muon events `0.482`
- median confidence when true two-muon is predicted as one muon `0.538`
- median confidence when true two-muon is predicted correctly `0.408`
- median confidence when true two-muon is predicted as three muons `0.552`

`exp_009_capacity_push_retuned`:

- true two-muon accuracy `0.272`
- median confidence for true two-muon events `0.465`
- median confidence when true two-muon is predicted as one muon `0.515`
- median confidence when true two-muon is predicted correctly `0.387`
- median confidence when true two-muon is predicted as three muons `0.541`

This is a strong sign that the middle class is not a compact cluster in the learned representation:

- the model is least confident when it predicts the two-muon class correctly
- it is often more confident when it pushes those same events to the neighboring classes

## Representative Events

Representative true two-muon events were chosen from the test split to cover:

- a detector-poor event pushed to one muon
- a mid-activity event predicted correctly
- a detector-rich event pushed to three muons

| category | category_label | source_event_index | target_class | active_hits | total_tot | time_span | z_span | total_energy_mc | primary_energy_mc | secondary_energy_mc | primary_off_vertical_deg_mc | exp008_predicted_class | exp008_confidence | exp009_predicted_class | exp009_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downward | True 2-muon predicted as 1 muon | 17327 | 1 | 26 | 733.000 | 1366.000 | 134.285 | 415.699 | 267.819 | 147.880 | 7.631 | 0 | 0.667 | 0 | 0.593 |
| middle | True 2-muon predicted as 2 muons | 6457 | 1 | 64 | 1564.000 | 2096.000 | 153.475 | 1127.908 | 699.798 | 428.110 | 53.299 | 1 | 0.420 | 1 | 0.396 |
| upward | True 2-muon predicted as 3 muons | 4723 | 1 | 248 | 6867.000 | 2600.000 | 163.489 | 9925.457 | 5153.635 | 4771.822 | 23.575 | 2 | 0.688 | 2 | 0.898 |

The representative event panel should be read as:

- left column: hit time vs `z`
- right column: hit geometry in the `x-y` plane
- point size and color scale with `TOT`

These examples are intended as qualitative checks of whether the detector response itself looks lower-count, middle, or upper-count.

## Verdict

Best-supported explanation at this stage:

- **primary bottleneck:** information limit in detector response
- **secondary factor:** some remaining model underfitting is still plausible because KM3Former beats the cheap detector-only baselines
- **not supported as the main issue:** a purely direction-driven failure mode

In experiment terms, the practical reading is:

- with only detector observables available, many true two-muon events do not produce a detector signature that is cleanly distinct from one- and three-muon events
- the MC-truth diagnostic does not change the inference setup; it only helps explain why the detector-only task is intrinsically hard

## Reproduction

```bash
uv run --with matplotlib python scripts/detector_overlap_analysis.py \
  runs/exp_008_capacity_push \
  runs/exp_009_capacity_push_retuned
```

