# mohler: primary — scores vs gold (0-5) on the test split (n=200)

95% percentile bootstrap CIs, 1000 resamples, seed 42. RMSE is on the 0-5 scale.

| Method | n | Pearson r [95% CI] | Spearman [95% CI] | RMSE [95% CI] |
|---|---|---|---|---|
| v1 | 200 | 0.464 [0.363, 0.558] | 0.473 [0.356, 0.575] | 0.939 [0.829, 1.052] |
| b0 | 200 | 0.464 [0.363, 0.558] | 0.473 [0.356, 0.575] | 0.939 [0.829, 1.052] |
| human ceiling: grader 1 vs grader 2 | 200 | 0.558 [0.438, 0.657] | 0.596 [0.478, 0.693] | 1.365 [1.180, 1.564] |

Each method's raw score is mapped to the gold scale by a straight line fitted on dev only (clipped to the scale); RMSE uses that mapped score, Pearson and Spearman the raw score. The ceiling row compares the two human graders with each other, not with their average.

# mohler: secondary — three labels on the test split (n=200)

95% percentile bootstrap CIs, 1000 resamples, seed 42.

| Method | n | QWK [95% CI] | Spearman [95% CI] | Accuracy [95% CI] |
|---|---|---|---|---|
| v1_cosine_tuned | 200 | 0.362 [0.229, 0.481] | 0.429 [0.304, 0.535] | 0.575 [0.505, 0.645] |
| b0_cosine_tuned | 200 | 0.362 [0.229, 0.481] | 0.429 [0.304, 0.535] | 0.575 [0.505, 0.645] |
| v1_cosine_app | 200 | 0.208 [0.129, 0.292] | 0.429 [0.304, 0.535] | 0.315 [0.255, 0.385] |
| b0_cosine_app | 200 | 0.359 [0.245, 0.460] | 0.429 [0.304, 0.535] | 0.590 [0.520, 0.655] |
| human ceiling: grader 1 vs grader 2 | 200 | 0.491 [0.375, 0.587] | 0.578 [0.458, 0.679] | 0.640 [0.575, 0.705] |

*_tuned: label cut-offs chosen on dev only (max QWK). *_app: the app's fixed cut-offs (50 / 80). With tuned cut-offs v1 and B0 give the same labels, because both are increasing functions of the cosine.

## Confusion matrix: v1_cosine_tuned

| | pred: incorrect | pred: partial | pred: correct |
|---|---|---|---|
| human: incorrect | 5 | 6 | 5 |
| human: partial | 18 | 15 | 26 |
| human: correct | 8 | 22 | 95 |

## Confusion matrix: b0_cosine_tuned

| | pred: incorrect | pred: partial | pred: correct |
|---|---|---|---|
| human: incorrect | 5 | 6 | 5 |
| human: partial | 18 | 15 | 26 |
| human: correct | 8 | 22 | 95 |

## Confusion matrix: v1_cosine_app

| | pred: incorrect | pred: partial | pred: correct |
|---|---|---|---|
| human: incorrect | 10 | 6 | 0 |
| human: partial | 30 | 27 | 2 |
| human: correct | 29 | 70 | 26 |

## Confusion matrix: b0_cosine_app

| | pred: incorrect | pred: partial | pred: correct |
|---|---|---|---|
| human: incorrect | 0 | 15 | 1 |
| human: partial | 0 | 42 | 17 |
| human: correct | 0 | 49 | 76 |
