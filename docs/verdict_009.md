**Verdict**

The safest reading is: the model is struggling mostly because many true 2-muon events do not form a clean, distinct detector signature. In the detector, they often look like a continuum between 1-muon-like events and 3-muon-like events, so the network has weak evidence for calling them “exactly 2.” That is what I mean by **information limit in detector response**. You can see that in [README.md](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/README.md), [detector_feature_overlap_test.csv](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/detector_feature_overlap_test.csv), and especially [detector_feature_cdfs.png](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/plots/detector_feature_cdfs.png).

The strongest evidence is that simple detector-activity features already order the classes. Features like `active_hits` and `total_tot` increase monotonically from 1 muon to 2 muons to 3 muons, but not strongly enough to cleanly separate 2 from its neighbors. Their pairwise AUCs are only moderate for adjacent classes: for `active_hits`, about `0.623` for `1 vs 2` and `0.686` for `2 vs 3`, while `1 vs 3` is easier at about `0.788`. That means the middle class is genuinely overlapped in hit-space. It is not an isolated cluster the model simply failed to learn.

The second clue is the baseline comparison in [baseline_metrics.json](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/baseline_metrics.json) and [confusion_comparison.png](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/plots/confusion_comparison.png). A cheap detector-only logistic model gets about `0.484` accuracy and two-muon recall about `0.243`. KM3Former improves that to about `0.532-0.539` accuracy and two-muon recall about `0.272-0.290`. So the transformer is learning something real, which means this is **not only** a data problem. But the improvement is modest, which says the data seen by the detector is already ambiguous.

The third clue is the diagnostic upper bound: when I let the cheap baseline use MC truth energies, accuracy jumps to about `0.751` and two-muon recall to about `0.639`. That is a huge gap. It says the generator-level truth contains multiplicity information that the detector-summary features do not expose cleanly. In plain terms: the event may truly have two muons at generation, but what reaches the detector can look much more like one or three in visible activity.

The confidence behavior strengthens that interpretation. In [model_ambiguity_summary.json](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/model_ambiguity_summary.json) and [class1_reliability_and_margin.png](/Users/djaz89/projects/KM3Former/runs/detector_overlap_analysis/plots/class1_reliability_and_margin.png), the model is actually **least confident when it predicts 2 muons correctly**. For `exp_009`, true 2-muon events predicted correctly have median confidence about `0.387`, but when the same true class is pushed to 1 muon or 3 muons the median confidence is higher, about `0.515` and `0.541`. That is a sign the network does not “see” a strong 2-muon prototype. It sees many events as belonging more naturally to one of the neighboring classes.

So the verdict broken down is:

- **Primary bottleneck: information limit in detector response.**
  - True 2-muon events overlap strongly with neighboring classes in detector-visible features.
  - Activity/light-yield features separate the extremes better than the middle.
- **Secondary bottleneck: some model underfitting is still possible.**
  - KM3Former does beat the cheap detector-only baselines.
  - So architecture/training still matters, just probably not enough to fully solve the problem.
- **Weaker evidence for label semantics mismatch as the main cause.**
  - It is still plausible that “generated multiplicity” and “detector-effective multiplicity” differ.
  - But from the current analysis alone, the cleanest statement is that the detector signature is ambiguous, regardless of whether that ambiguity comes from propagation losses, geometry, or bundle structure.

What this verdict does **not** mean is “the model is bad” or “2-muon events are impossible.” It means the current target may be asking for a finer distinction than the detector response cleanly supports with the information we currently expose. If we want to push further, the best next step is probably not just “make the transformer bigger,” but to add features or truth-aligned labels that better reflect **detector-effective multiplicity**.