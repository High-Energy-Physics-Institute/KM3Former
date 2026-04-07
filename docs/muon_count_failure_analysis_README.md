# Muon Count Failure Analysis

## Scope

This note summarizes the error analysis for the current muon-count models, with special focus on the middle class, which is the working proxy for the difficult two-muon category.

Working interpretation used in this report:

- class `0` -> lower-count neighbor
- class `1` -> middle class, treated here as the two-muon hypothesis
- class `2` -> upper-count neighbor

The report is based on two runs:

- [exp_008_capacity_push](/Users/djaz89/projects/KM3Former/runs/exp_008_capacity_push)
- [exp_009_capacity_push_retuned](/Users/djaz89/projects/KM3Former/runs/exp_009_capacity_push_retuned)

The joined event tables were produced from:

- saved prediction tensors in each run directory
- raw Monte Carlo labels and energies from [muon_data_7224_7247.h5](/Users/djaz89/projects/KM3Former/data/muon_data_7224_7247.h5)
- deterministic split reconstruction from [metadata.json](/Users/djaz89/projects/KM3Former/data/metadata.json)

Relevant scripts:

- [analyze_muon_count_predictions.py](/Users/djaz89/projects/KM3Former/scripts/analyze_muon_count_predictions.py)
- [plot_muon_count_failure_modes.py](/Users/djaz89/projects/KM3Former/scripts/plot_muon_count_failure_modes.py)

## Artifacts

### `exp_008_capacity_push`

- [overview plot](/Users/djaz89/projects/KM3Former/runs/exp_008_capacity_push/eda_plots/test_predictions_joined_focus_class_1_overview.png)
- [scatter plot](/Users/djaz89/projects/KM3Former/runs/exp_008_capacity_push/eda_plots/test_predictions_joined_focus_class_1_scatter.png)
- [summary json](/Users/djaz89/projects/KM3Former/runs/exp_008_capacity_push/eda_plots/test_predictions_joined_focus_class_1_summary.json)
- [joined csv](/Users/djaz89/projects/KM3Former/runs/exp_008_capacity_push/test_predictions_joined.csv)

### `exp_009_capacity_push_retuned`

- [overview plot](/Users/djaz89/projects/KM3Former/runs/exp_009_capacity_push_retuned/eda_plots/test_predictions_joined_focus_class_1_overview.png)
- [scatter plot](/Users/djaz89/projects/KM3Former/runs/exp_009_capacity_push_retuned/eda_plots/test_predictions_joined_focus_class_1_scatter.png)
- [summary json](/Users/djaz89/projects/KM3Former/runs/exp_009_capacity_push_retuned/eda_plots/test_predictions_joined_focus_class_1_summary.json)
- [joined csv](/Users/djaz89/projects/KM3Former/runs/exp_009_capacity_push_retuned/test_predictions_joined.csv)

## What The Plot Data Means

The plots use only test-set events whose true label is class `1`, then split those events by the model prediction: predicted `0`, predicted `1`, or predicted `2`.

The variables come from two sources:

- Monte Carlo truth from the HDF5 file:
  - `primary_energy`
  - `secondary_energy`
  - `total_energy_first_two = primary_energy + secondary_energy`
  - `primary_off_vertical_deg`
  - `secondary_energy_fraction = secondary_energy / total_energy_first_two`
- detector-response proxy from the raw hit tensor:
  - `active_hits`: number of hits with `TOT > 0`
  - `total_tot`: sum of positive TOT values over the event

Important limitation:

- the checked-in HDF5 file stores two energy slots and two direction triplets, so this analysis can test whether the middle class behaves more like the lower or upper neighbor in visible activity, but it does not fully resolve every muon separately for higher-multiplicity events

## How To Read The Figures

### Overview plot

The overview figure has six panels:

- confusion matrix:
  - all test events
  - rows are true classes, columns are predicted classes
  - this shows that the middle class is the weakest class overall
- total energy histogram:
  - only true class `1`
  - compares `log10(total_energy_first_two)` for predictions `0`, `1`, and `2`
- active hits histogram:
  - only true class `1`
  - compares the number of hits with `TOT > 0`
- total TOT histogram:
  - only true class `1`
  - compares `log10(total_tot)`
- direction histogram:
  - only true class `1`
  - compares the primary off-vertical angle in degrees
- energy sharing histogram:
  - only true class `1`
  - compares `secondary_energy_fraction`

Interpretation rule:

- if the three curves separate strongly, that variable is informative for why the model pushes the event toward the lower or upper class
- if the curves overlap heavily, that variable is probably not the main driver of the failure mode

### Scatter plot

The scatter figure has two panels:

- energy plane:
  - x-axis: primary MC energy
  - y-axis: secondary MC energy
  - both axes are logarithmic
  - each point is a true class `1` event colored by model prediction
- energy vs detector activity:
  - x-axis: total MC energy of the first two muons
  - y-axis: active hit count
  - x-axis is logarithmic
  - this directly tests whether events that look like lower-count or upper-count classes differ in visible detector activity

## Main Findings

The same pattern appears in both runs.

The middle class is not concentrated around a single compact detector-response regime. Instead, it spreads along a continuum:

- low-activity, low-energy middle-class events are often pushed down to class `0`
- high-activity, high-energy middle-class events are often pushed up to class `2`
- the correctly predicted middle-class events sit between those two regimes

This is easiest to see in activity and energy, not in direction.

## Quantitative Summary

### `exp_008_capacity_push`

True class `1` support: `699`

- predicted `0`: `319` events, `45.6%`
- predicted `1`: `203` events, `29.0%`
- predicted `2`: `177` events, `25.3%`

Median values for true class `1`:

| Predicted class | Median total energy | Median active hits | Median total TOT | Median off-vertical angle |
| --- | ---: | ---: | ---: | ---: |
| `0` | `897.1` | `52.0` | `1303.0` | `36.49 deg` |
| `1` | `1096.7` | `65.0` | `1640.0` | `30.76 deg` |
| `2` | `1655.1` | `89.0` | `2286.0` | `28.92 deg` |

Reading this table:

- the class-`0` mistakes are visibly quieter events
- the class-`2` mistakes are visibly busier events
- the correct class-`1` predictions sit in the middle

### `exp_009_capacity_push_retuned`

True class `1` support: `699`

- predicted `0`: `299` events, `42.8%`
- predicted `1`: `190` events, `27.2%`
- predicted `2`: `210` events, `30.0%`

Median values for true class `1`:

| Predicted class | Median total energy | Median active hits | Median total TOT | Median off-vertical angle |
| --- | ---: | ---: | ---: | ---: |
| `0` | `878.8` | `51.0` | `1277.0` | `36.12 deg` |
| `1` | `1078.4` | `65.0` | `1645.0` | `33.22 deg` |
| `2` | `1544.4` | `87.5` | `2196.5` | `29.04 deg` |

This second run shows the same ordering as `exp_008`, which makes the conclusion more robust:

- energy and detector activity shift strongly with the predicted class
- direction changes only moderately
- secondary-energy fraction is fairly similar across the three prediction groups and is not the dominant separator

## Scientific Interpretation

The current evidence supports the following working explanation:

- many true middle-class events do not look like a single clean detector signature
- some of them look detector-poor and are classified like the lower-count class
- some of them look detector-rich and are classified like the upper-count class

That is consistent with the hypothesis that the generator-level multiplicity is not the same as detector-effective multiplicity. In other words, a simulated extra muon may exist at the generation surface, but the detector response may resemble a lower-multiplicity event if that extra component contributes little visible activity.

At the same time, the upward confusions show the opposite side of the same effect:

- some true middle-class events are energetic and hit-rich enough that the network treats them as upper-class events

So the failure mode looks more like a detector-visibility continuum than a clean direction-based separation problem.

## Caveats

- this analysis assumes the current working interpretation that class `1` is the middle two-muon category
- the HDF5 auxiliary truth is limited to two energy slots and two direction triplets
- the direction fields for the first two slots appear highly redundant in this file, so they may represent bundle-level direction rather than rich per-muon geometry
- no propagation or distance-to-detector variable is currently stored in the preprocessed tensors, so this report cannot directly prove that a generated muon failed to reach the instrumented volume

## Next Steps

- add event-level features that better represent detector-effective multiplicity, such as active DOM count, hit topology, and longitudinal light profile
- if available upstream, attach generator-to-detector geometry variables such as closest approach, cylinder entry point, or propagated survival information
- compare the same plots for true class `0` and true class `2` so the middle class can be framed relative to both neighbors in the same detector-response space
