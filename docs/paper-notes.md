# Paper notes: Do-PFN

**Do-PFN: In-Context Learning for Causal Effect Estimation.** Robertson, Reuter, Guo, Hollmann, Hutter,
Schölkopf. NeurIPS 2025, arXiv:2506.06039. PDF: `docs/papers/Do-PFN.pdf`.

These notes summarize the paper for implementation. They are valid across all milestones.

## The idea

- **Task.** Given an *observational* tabular dataset of treatment $t \in \{0,1\}$, covariates $x$ and
  outcome $y$, predict the outcome under an *intervention* $do(t)$ for new query points, i.e. the
  conditional interventional distribution (CID) $p(y \mid do(t), x)$. From it, compute the CATE
  $\tau(x) = E[y \mid do(1), x] - E[y \mid do(0), x]$.
- **Why it's hard.** A regression fitted on observational data learns $p(y \mid t, x)$, which differs
  from $p(y \mid do(t), x)$ whenever there are confounders, mediators or hidden variables.
- **Method.** Do-PFN is a TabPFN-style transformer (prior-data fitted network). It is pre-trained on
  millions of synthetic datasets drawn from random structural causal models (SCMs). In each training
  step, the context is observational data from an SCM, and the targets are *interventional* outcomes
  from the same SCM. The model thereby meta-learns to do causal adjustment in context, without being
  given the causal graph.

## Algorithm 1 (one pre-training step)

1. Sample a random SCM $\psi$ from the prior.
2. Sample $M_{ob} \sim \text{Uniform}\{M_{min}, \dots, M_{max}\}$ observational rows $(t, x, y)$ by
   running the SCM normally.
3. Set $M_{in} = M_{max} - M_{ob}$. For each query row $k$:
   - sample a fresh noise vector $\epsilon_k$,
   - compute the covariates $x^{in}_k$ from the **unintervened** SCM with that noise
     ("pre-treatment" values),
   - sample $t^{in}_k$ from a prior over treatment values (binary),
   - apply $do(t^{in}_k)$ (cut incoming edges into the treatment, fix its value) and compute
     $y^{in}_k$ in the intervened SCM **with the same noise** $\epsilon_k$ (descendants of $t$,
     e.g. mediators, are recomputed).
4. Loss is $\sum_k -\log q_\theta(y^{in}_k \mid do(t^{in}_k), x^{in}_k, D_{ob})$, then one gradient
   step (Adam, mini-batches of datasets).

**Subtlety in step 3.** If a covariate is a mediator (downstream of $t$), the model is conditioned on
its factual pre-intervention value, while $y^{in}$ uses the recomputed mediator. The team still needs
to confirm this choice (see `docs/decisions.md`, D5).

## Prior details (paper Appendix C, D.1)

- Random DAG via topological ordering. Number of nodes varies (v1 up to 10, v1.1 up to 60).
- Additive noise mechanisms $z_k = \gamma(\sum_{l \in PA(k)} w_l z_l) + \epsilon_k$.
- Nonlinearity $\gamma$ sampled uniformly from $\{x^2, \text{ReLU}, \tanh\}$ per node.
- Weights Kaiming-uniform, $w_l \sim U(-1/\sqrt{|PA(k)|}, 1/\sqrt{|PA(k)|})$.
- Root noise std $\sigma_{exo} \sim U[1,3]$. Additive noise std $\sigma_\epsilon = 0.3 \beta$,
  $\beta \sim \text{Beta}(1,5)$. Gaussian noise.
- Dataset size $M_{min} = 10$, $M_{max} = 2200$.
- **Caution:** Table 1 in the paper has errors in the case-study equations. Trust Figure 2 and the
  authors' code over Table 1.

## Model

- TabPFN architecture, 7.3M parameters. The only change is an indicator telling the model that
  column 0 is the treatment.
- Output head is a **bar distribution** (histogram over bins of $y$), trained with NLL.
  Do-PFN v1 trained 48h on one RTX 2080, v1.1 96h.
- CATE is the mean of the predicted distribution under $do(1)$ minus the mean under $do(0)$.

## Evaluation in the paper

- Six synthetic case studies: Observed Confounder, Observed Mediator, Confounder + Mediator,
  Unobserved Confounder (unidentifiable), Back-Door Criterion, Front-Door Criterion. Plus Common
  Effect (randomized treatment), where plain regression matches Do-PFN.
- Metric is MSE normalized by the range of $y$:
  $\frac{1}{n}\sum_i \left(\frac{y_i - \hat y_i}{\max(y) - \min(y)}\right)^2$.
- **Dont-PFN** is the key control: same model and prior, trained on *observational* outcomes of the
  query points.
- CATE baselines: S/T/X-learners, DML, causal forest, TARNet, DragonNet, DoWhy with the true graph.
  Also RealCause and two known-graph datasets (Law School, Amazon Sales).
- Stated limitations: depends on the synthetic prior matching reality, binary treatment only, no
  truly real ground truth.
