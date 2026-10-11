# mohler: primary — scores vs gold (0-5) on the test split (n=200)

95% percentile bootstrap CIs, 1000 resamples, seed 42. RMSE is on the 0-5 scale.

| Method | n | Pearson r [95% CI] | Spearman [95% CI] | RMSE [95% CI] |
|---|---|---|---|---|
| v1 | 200 | 0.446 [0.338, 0.542] | 0.448 [0.333, 0.554] | 0.950 [0.840, 1.067] |
| b0 | 200 | 0.446 [0.338, 0.542] | 0.448 [0.333, 0.554] | 0.950 [0.840, 1.067] |
| human ceiling: grader 1 vs grader 2 | 200 | 0.558 [0.438, 0.657] | 0.596 [0.478, 0.693] | 1.365 [1.180, 1.564] |

Each method's raw score is mapped to the gold scale by a straight line fitted on dev only (clipped to the scale). Pearson and Spearman do not depend on that map; RMSE does. The ceiling row compares the two human graders with each other, not with their average.

# mohler: secondary — three labels on the test split (n=200)

95% percentile bootstrap CIs, 1000 resamples, seed 42.

| Method | n | QWK [95% CI] | Spearman [95% CI] | Accuracy [95% CI] |
|---|---|---|---|---|
| v1_cosine_tuned | 200 | 0.356 [0.231, 0.476] | 0.413 [0.298, 0.522] | 0.590 [0.525, 0.655] |
| b0_cosine_tuned | 200 | 0.356 [0.231, 0.476] | 0.413 [0.298, 0.522] | 0.590 [0.525, 0.655] |
| v1_cosine_app | 200 | 0.193 [0.118, 0.277] | 0.413 [0.298, 0.522] | 0.310 [0.250, 0.375] |
| b0_cosine_app | 200 | 0.337 [0.229, 0.432] | 0.413 [0.298, 0.522] | 0.575 [0.505, 0.640] |
| human ceiling: grader 1 vs grader 2 | 200 | 0.491 [0.375, 0.587] | 0.578 [0.458, 0.679] | 0.640 [0.575, 0.705] |

*_tuned: label cut-offs chosen on dev only (max QWK). *_app: the app's fixed cut-offs (50 / 80). With tuned cut-offs v1 and B0 give the same labels, because both are increasing functions of the cosine.

## Confusion matrix: v1_cosine_tuned

| | pred: incorrect | pred: partial | pred: correct |
|---|---|---|---|
| human: incorrect | 2 | 9 | 5 |
| human: partial | 10 | 27 | 22 |
| human: correct | 2 | 34 | 89 |

## Confusion matrix: b0_cosine_tuned

| | pred: incorrect | pred: partial | pred: correct |
|---|---|---|---|
| human: incorrect | 2 | 9 | 5 |
| human: partial | 10 | 27 | 22 |
| human: correct | 2 | 34 | 89 |

## Confusion matrix: v1_cosine_app

| | pred: incorrect | pred: partial | pred: correct |
|---|---|---|---|
| human: incorrect | 11 | 5 | 0 |
| human: partial | 31 | 26 | 2 |
| human: correct | 34 | 66 | 25 |

## Confusion matrix: b0_cosine_app

| | pred: incorrect | pred: partial | pred: correct |
|---|---|---|---|
| human: incorrect | 0 | 15 | 1 |
| human: partial | 0 | 42 | 17 |
| human: correct | 0 | 52 | 73 |
