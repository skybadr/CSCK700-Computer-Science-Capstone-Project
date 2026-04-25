# Design Notes

A running log of design decisions made during the project. Each entry should
capture the decision, the alternatives considered, and the rationale. This
becomes raw material for the methodology chapter of the dissertation.

---

## 2026-04-25 — Initial scaffold

**Decision:** Use a vertical-slice strategy (run 20 prompts end-to-end first)
rather than the proposal's waterfall (lit review → design → dataset →
framework → experiments).

**Rationale:**
- Surfaces hidden problems early (Arabic tokenisation, BERTScore quality,
  LLMLingua behaviour on Arabic) while there's time to react.
- Provides a working artefact from week 1, useful psychologically and as
  a regression check.

**Alternatives considered:**
- Strict waterfall as in the proposal — rejected because the project is
  already 2–3 weeks behind and waterfall delays integration risk.

---

## TODO: decisions to make

- [ ] **LLM for benchmarking.** Candidates: `gpt-4o-mini`, `claude-haiku-4-5`,
      open Arabic-capable model (Jais, AceGPT). Default in the scaffold is
      `gpt-4o-mini` for cost; revisit before scaling.
- [ ] **BERTScore model for Arabic.** Default is `bert-base-multilingual-cased`.
      Should also evaluate with `aubmindlab/bert-base-arabertv02` and report
      both — the choice affects all downstream conclusions.
- [ ] **LLMLingua version.** LLMLingua-2 (2024) is meaningfully different from
      the original. Need to decide whether to benchmark both or pick one and
      justify.
- [ ] **Target compression ratios.** Single ratio (e.g. 0.5) for all
      experiments, or sweep across {0.3, 0.5, 0.7}? Sweep is more thorough
      but multiplies cost.
- [ ] **Dataset size split.** Proposal says 800–1200, with 80/20 dev/test.
      Consider tiered design: 1200 for cheap metrics, 200-prompt subset for
      full LLM-output evaluation (cost-controlled).
- [ ] **Dialect inclusion.** Proposal mentions "limited inclusion of
      dialectal Arabic". Need to scope concretely: which dialect(s),
      what proportion, and how to handle their effect on BERTScore.
