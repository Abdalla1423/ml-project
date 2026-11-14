# Resources

## Code
- **Do-PFN** (paper code: pre-trained model, prior, case studies): https://github.com/jr2021/Do-PFN
- **Do-PFN-prior** (starting point for our prior: simple configurable SCM prior compatible with
  nanoTabPFN; only generates observational data, no designated treatment):
  https://github.com/oossen/Do-PFN-prior
- **nanoTabPFN** (Pfefferle et al., minimal TabPFN with a scikit-learn-like interface):
  https://github.com/automl/nanoTabPFN
- **TFM-Playground** (nanoTabPFN training loop): https://github.com/automl/TFM-Playground

Do-PFN and Do-PFN-prior are copied into `Do-PFN/` and `Do-PFN-prior/` (see `docs/third-party.md` for commits
and changes). Code from the other repos may be copied in the same way and adapted as needed (decision
D7). Do not assume how a repo is structured. Inspect it before relying on any API.

## Literature
- Robertson et al., "Do-PFN: In-Context Learning for Causal Effect Estimation", NeurIPS 2025,
  arXiv:2506.06039 (`docs/papers/Do-PFN.pdf`).
- Müller et al., "Transformers Can Do Bayesian Inference", ICLR 2022.
- Peters, Janzing, Schölkopf, "Elements of Causal Inference", MIT Press 2017.
- PyTorch docs: https://pytorch.org/docs/stable/index.html
- scikit-learn docs: https://scikit-learn.org/stable/documentation.html

## Submission
- Report: via ISIS, course LaTeX template, max 10 pages, no code.
- Code: GitHub repo shared with `marvinsxtr`, submitted commit tagged `msN`, runnable on the cluster.

## Infrastructure
- Supervisor GitHub handle: `marvinsxtr` (needs access to the repo).
- Cluster datasets: `/home/space/datasets/<dataset>`, made readable with `chmod -R a+r`.
