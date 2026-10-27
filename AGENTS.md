# Working in this repository

SparkSeek is SparkGLM's and SparkQwen's distribution model applied to
DeepSeek-V4.1-Flash-1017: Atlas on DGX Sparks, one entry point, a pinned
engine, receipts for every claim. Four Sparks serve DeepSeek's release as
published; two Sparks serve a lossy 2.5 bpw pack.

- Keep it simple: one entry point, one install directory, one results tree.
  Prefer removing code to adding it.
- Engine changes belong in `Enntity/atlas`, never here: upstream-worthy work
  on `upstream/deepseek-v41-flash-*` (cut from Atlas-Inf `main`), SparkSeek-only
  work on `sparkseek/*`. Then update the pin in `install/atlas-source.json`
  (commit and tree together) and `docs/ENGINE.md`. Never push to Atlas-Inf.
- This repository holds accepted work only. Plans, logs, WIP and host-specific
  notes go to a private notes repository. Keep host names, user names, private
  addresses and home paths out of everything here except unedited receipts in
  `results/*/raw/`.
- A performance or capacity claim needs receipts and `SHA256SUMS` under
  `results/`; a performance baseline is measured on the same Sparks. Report
  every repetition and say which opt-ins were on.
- Keep lossy or quality-affecting options off by default, and name the TP2
  pack lossy wherever it appears.
- Never take Sparks that serve production without their owner's go-ahead;
  reserve shared hardware through LLooM.
- New files carry an SPDX header. Credit ideas and code taken from other
  projects (origin, license) in code and docs.
- Before sending a change, run the checks in `.github/workflows/static.yml`.
