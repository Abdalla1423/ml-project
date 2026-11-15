# Material for the ms1 report

Created by `python scripts/make_report_material.py` from the cluster results in `../results/`
(run `run2`). Every figure exists as `.png` and `.pdf`; tables are LaTeX (`booktabs` package) and
can be included with `\input{...}`. `tab_results.md` is the results table for quick reading.

Note on the numbers: `tab_results.tex` holds the cluster evaluation (`../results/metrics.json`).
The box plot and the win rates use scores per dataset that were recomputed on a laptop with the same
checkpoint (`per_dataset_scores.csv`). The case-study files are identical on both machines; the 256
held-out prior datasets are not (the same seed gives slightly different datasets on another machine
and PyTorch version), so prior numbers differ in the third decimal.

## Figures

| File | Section | Suggested caption |
|---|---|---|
| `fig_prior_examples` | Exploration | 3 datasets sampled from our prior. Left: the sampled causal graph with treatment `t` and outcome `y`. Middle: observational data, colored by treatment. Right: interventional outcomes of the query rows under do(t). Horizontal axis: first covariate. |
| `fig_training` | Results, overfitting | Left: training and validation NLL during pre-training (30000 steps, 92 min). Right: CATE error on the validation datasets, compared with the difference of group means. |
| `fig_nmse_boxplots` | Results | Normalized MSE per dataset (log scale) on held-out prior data and the six case studies, for the interventional outcome (top) and the CATE (bottom). Boxes: quartiles, line: median. |
| `fig_predictions` | Results | Predicted interventional outcomes of the trained model (crosses) on top of the true interventional data, for the datasets of the first figure. |
| `../results/case_study_metrics.png` | Results (alternative) | Mean normalized MSE with 95 % bootstrap intervals. The Observed Mediator panel is stretched by one dataset; prefer the box plot. |

## Tables

| File | Section | Content |
|---|---|---|
| `tab_prior_stats.tex` | Exploration, data quality | What the prior generates (1000 sampled datasets). |
| `tab_setup.tex` | Method | Model and training setup. |
| `tab_results.tex` | Results | Mean and median normalized MSE per dataset and model; best value in bold. |
| `tab_win_rates.tex` | Results | Share of datasets on which our model beats both baselines. |

## Key numbers for the text

Training
- 30000 steps of 16 datasets in 92 minutes on one GPU; the model has 1.09 M parameters.
- Validation NLL falls from 1.43 to 1.070; over the last 3000 steps it changes by 0.0005.
- Training NLL at the end is 1.06, validation 1.07: no gap, so no overfitting. Every training step uses freshly sampled datasets, so the model never sees a dataset twice.
- Validation CATE MSE (normalized outcome): 0.141, difference of group means: 0.203.

Held-out prior data (256 datasets, cluster evaluation)
- Outcome NMSE: our model 0.0182, regression 0.0272, group means 0.0274.
- CATE NMSE: our model 0.0047, regression 0.0209, group means 0.0082.
- Our model is better than both baselines on 79 % of the datasets for the outcome and 61 % for the CATE.

Case studies (100 datasets each)
- Outcome: lowest mean error in five of six case studies (all except Observed Mediator).
- CATE: lowest mean error in four of six. Not on Unobserved Confounder (0.0476 vs 0.0441 for group means; the effect is not identifiable there) and not on Observed Mediator.
- Per dataset (`tab_win_rates.tex`): our model beats both baselines at once on 46–53 % of the datasets of a case study for the outcome and on 24–43 % for the CATE (Observed Mediator: 26 % and 15 %). The lower mean error therefore comes from fewer large errors, not from winning on every dataset.
- Observed Mediator: the difference of group means is the right estimator (no confounding) and is best. Our mean error (90.4) comes from one dataset whose interventional outcomes are almost constant (range 0.002), so dividing by the squared range inflates the error; the median is 0.0191 (group means 0.0100).

What the prior generates (1000 datasets, `tab_prior_stats.tex`)
- A common cause of treatment and outcome exists in 26 % of the datasets, and in 1 % at least one is hidden.
- A mediator exists in 84 %; in 83 % one is among the covariates.
- In 0 % there is no causal path from treatment to outcome (true effect 0).
- Data quality: 0 % of the contexts contain only one treatment group, 0 % have a group below 5 %, and 0 % have no covariates at all.

Limits to state
- Results come from one training run with one seed.
- A fixed seed reproduces the data on one machine, not across machines or PyTorch versions.
- The mean normalized MSE is sensitive to datasets with a near-constant outcome; medians are in the table.
- CATE baselines of the paper (S/T/X-learner, DML, ...) and Dont-PFN are not part of ms1.
