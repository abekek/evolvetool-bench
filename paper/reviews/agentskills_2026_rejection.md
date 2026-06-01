# EvolveTool-Bench — AgentSkills 2026 Workshop Review

**Venue:** ACM CAIS 2026 Workshop AgentSkills
**Submission #:** 35
**Authors:** Alibek T Kaliyev, Artem Maryanskyy
**Submitted:** 30 Apr 2026 (modified: 17 May 2026)
**Decision:** **Reject** (Program Chairs, 14 May 2026)

**Paper title:** EvolveTool-Bench: Evaluating the Quality of LLM-Generated Tool Libraries as Software Artifacts

**Abstract (as submitted):**
> Modern LLM agents increasingly create their own tools and skills at runtime, from Python functions to API clients, yet existing benchmarks evaluate them almost exclusively by downstream task completion. This is analogous to judging a software engineer only by whether their code runs, ignoring redundancy, regression, and safety. We introduce EvolveTool-Bench, a diagnostic benchmark that evaluates LLM-generated tool and skill libraries as software artifacts. Across three domains requiring actual code execution (proprietary data formats, API orchestration, and numerical computation), we define library-level software quality metrics (reuse, redundancy, composition, regression, and utilization) alongside a per-tool Tool Quality Score measuring correctness, robustness, generality, and code quality. Evaluating four systems spanning the skill-evolution spectrum across 99 tasks and three models (Claude Sonnet 4, Claude Haiku 4.5, and GPT-4o), we show that on Claude models, systems with near-identical task completion (63–68%) differ by up to 13 percentage points in library health, a gap invisible to task-only evaluation. Across providers, discriminative power varies: GPT-4o converges to identical low performance regardless of evolution strategy, suggesting provider-specific interactions that benchmark designers must account for. Per-dimension analysis reveals that correctness, not code style, is the frontier: hidden tests catch defects that self-generated tests miss. We release the benchmark as an open evaluation framework for any tool-generating or skill-generating system.

---

## Review 1 — Reviewer 5Wxk

**Title:** A Useful Diagnostic Framework, with Open Validity Questions
**Date:** 13 May 2026, 03:34 (modified 13 May 2026, 09:00)
**Rating:** **6: Marginally above acceptance threshold**
**Confidence:** 4 (confident but not absolutely certain)

### Summary
The paper proposes EvolveTool-Bench, a diagnostic benchmark that treats LLM-generated tool/skill libraries as software artifacts. It defines a per-tool Tool Quality Score (correctness, robustness, generality, code quality) and six Library Health sub-metrics (reuse, redundancy, quality gate, utilization, composability, regression), combined into a composite EvolveTool Score. Four systems spanning a no-evolution → unvalidated → strategy-only → validated-code-evolution spectrum are evaluated on 99 tasks across three domains and three models. The headline finding is that Claude configurations cluster in a narrow TC band (63–68%) but spread 13pp in LH, demonstrating that task completion alone is insufficient to evaluate skill-generating systems.

### Strengths
- **Multi-dimensional diagnostic framework.** The three-tier metric design (TQS / LH / ETS) is a genuine conceptual advance over single-number task-success evaluation. The decomposition of reuse into correct reuse vs. incorrect reuse in particular surfaces a non-obvious failure mode (the "reuse paradox": 72.2% overall reuse, 51.9% of which is incorrect) that aggregate metrics would have hidden.
- **Structured-session design enables causal analysis.** Fixing each session to 11 tasks across six categories (seed / gap / variant / compose / regress / adversarial) with known dependencies makes it possible to ask why a system succeeded or failed, not just whether it did. The per-task-type breakdown in Table 7 is a direct payoff.
- **Methodological honesty.** The paper actively surfaces uncomfortable findings — Code-Evol scoring lower than No-Evolution on gap (−5.5pp) and variant (−16.7pp) tasks, the reuse paradox, full GPT-4o convergence, Q-metric near-saturation — and includes an unusually detailed 8-item Limitations section. This transparency substantially raises the work's credibility.

### Weaknesses
**W1. The predictive validity of Library Health is not empirically established.**
The paper's central claim is that LH measures a distinct, valuable quality dimension that TC misses. The framing is reasonable, but the paper does not demonstrate that LH predicts anything: higher-LH libraries do not produce better task completion in the data (in fact Code-Evol with LH=51.6% scores lower on gap and variant than No-Evolution with LH=41.4%). A natural validity check — does LH measured in early sessions predict TC in later sessions, or in deployment-style follow-up tasks? — is not performed. Without such evidence, LH is at best a descriptive diagnostic, not a predictively meaningful quality signal, and the case for benchmark designers to optimize for it remains unsupported.

**W2. The benchmark's discriminative power depends on the representativeness of evaluated systems.**
The four systems span "no evolution → unvalidated → strategy-only → validated code evolution." Three of these are weak by construction: One-Shot has no validation by definition, No-Evolution creates no tools, and Strategy-Only is admitted in Section 4 to be "not a faithful reimplementation of EvoSkill's full system." The 13pp LH spread the paper headlines may therefore reflect a manufactured spectrum rather than differences among genuinely competing tool-evolving systems. Comparing against, e.g., the full EvoSkill framework, a Voyager-style Python skill library, or TOOLMAKER (Wang et al., 2025) — none of which appear in the evaluation — would substantially strengthen the claim that the benchmark differentiates systems that practitioners would actually deploy.

**W3. The hidden test suite is not truly independent, not sized, and not powered.**
Three concerns compound here:
- **Author independence:** the proprietary binary formats (ABR, RLE, VDL, QLOG, TPACK) were authored by the same team that wrote the hidden tests. "Hidden from the system" ≠ "independent of the authors." A genuine independence guarantee would require third-party-written tests or public formats with established conformance suites.
- **Test counts are never reported:** the paper does not state how many hidden unit tests, adversarial inputs, or held-out generality inputs exist per tool. With per-tool correctness averaging 0.17–0.23, the underlying binomial sample sizes determine whether these are estimates of true correctness or noisy snapshots of one bug pattern.
- **Statistical power is asymmetric:** only Code-Evol/Sonnet is run with n=4; every other configuration is n=1. Given known LLM run-to-run variance at temperature 1.0, the headline 13pp LH spread cannot be assessed for significance without multi-seed runs on the comparison configurations.

### Additional Concerns
- ETS weights are chosen to support the thesis (LH=0.30 "reflecting the benchmark's thesis"). The robustness check in Table 2 perturbs weights but never tests LH=0, the relevant stress test.
- GPT-4o convergence is attributed to "harness interaction" without diagnosis. If the benchmark only discriminates on Claude, its provider-agnostic framing weakens.
- Two sub-metrics are admitted not to discriminate: Q (code quality, near-saturated) and γ (composability, only 9 compose tasks total). Both still receive equal weight in LH.
- Domain B (API orchestration) has only 1 session (N=11 tasks), yet per-domain claims are drawn.
- **Missing literature:** the program-synthesis library-learning lineage (DreamCoder, Stitch, Babble, AbstractBeam, Leroy), which already evaluates library quality via compression metrics, is engaged with only at the level of a single citation. Concurrent work on tool/skill benchmarks (EvoSkills, arXiv:2604.01687; TOOLMAKER/TM-Bench, arXiv:2502.11705) is not cited.
- **Disclosure:** Code-Evol corresponds to a publicly available open-source system by an author of this paper. This does not invalidate the benchmark contribution — the paper's central claim is about discriminative power, not Code-Evol's superiority — but it should be disclosed in camera-ready.

### Recommendation
A solid workshop paper introducing a conceptually meaningful framework, presented with unusual methodological honesty. The headline empirical claim, however, rests on weak ground: the predictive value of LH is asserted rather than shown, the comparison systems are partly straw, and the hidden test suite (the benchmark's main epistemic guarantee) is undersized, undisclosed in count, and authored by the same team that designed the targets it measures. A revision that adds an LH-predicts-future-TC validity check, evaluates at least one competitive full-strength baseline (e.g., complete EvoSkill or TOOLMAKER), and documents the hidden-test suite (sizes, design protocol, ideally third-party authorship) would be substantially stronger.

---

## Review 2 — Reviewer oR4e

**Date:** 11 May 2026, 21:25 (modified 13 May 2026, 09:00)
**Rating:** **5: Marginally below acceptance threshold**
**Confidence:** 4 (confident but not absolutely certain)

### Summary
In this paper, authors present a diagnostic benchmark for assessing LLM-generated tool and skill libraries not only as collections of functions on data but also treated solely as software artifacts to be evaluated. New tool creating agents are often evaluated primarily across a variety of task domains, with limited consideration of library level failure modes such as redundancy, incorrect reuse, regression (which is not just at the point of introduced examples), low utilization and poor hidden test performance; this work argues for a more diverse perspective. The benchmark consists of structured 11-task sessions, including seed, gap, variant, compose, regress and adversarial tasks. It assesses four systems — No Evolution, One Shot, Strategy Only and Code Evol over three domains: data transformation, API orchestration and numerical computation. The core empirical claim is that completing tasks on your own conceals generalizable differences in library quality. The paper also reports a reuse paradox, where Code Evol achieves high reuse but much of that reuse is incorrect because defective tools are reused. I think the problem is important and very relevant to the Agent Skills workshop. The core idea, that generated skill libraries should be evaluated as software artifacts, is strong. However, I am not fully convinced by the current paper. The benchmark design is promising, but the experiments and reporting have enough issues that I would place the paper slightly below the acceptance threshold in its current form.

### Strengths
- The motivation is strong. If agents are going to create and accumulate tools over time, then evaluating only final task success is clearly incomplete. A generated tool library can become brittle, redundant, underused, or regression prone even when some tasks are solved.
- The structured task design is one of the best parts of the paper. The seed, gap, variant, compose, regress, and adversarial categories are a good way to diagnose different library behaviors. This is more useful than a flat task completion benchmark.
- The distinction between per tool quality and library health is also useful. Correctness, robustness, generality, code quality, reuse, duplication, utilization, composability, and regression stability are all relevant dimensions for tool library evaluation.
- The reuse paradox is an interesting result. The paper shows that high reuse is not necessarily good if the reused tools are defective. That is a useful takeaway for future agent skill benchmarks.
- The paper is also fairly transparent about limitations, including the small number of multi seed runs, the GPT 4o harness issue, the weak code quality metric, and the post hoc reuse decomposition.

### Weaknesses
- **Underspecified metric edge cases.** No Evolution and Strategy Only create zero executable tools, but still receive Library Health and ETS scores. It is unclear how quality gate, utilization, duplication, mean TQS, and safety score are computed when the generated library is empty. This is important because these baselines are central to the paper.
- **Table 2 / Table 3 inconsistency.** Table 2 reports one set of ETS values for the Sonnet systems under the original weighting, while Table 3 reports different ETS values for what look like the same systems. If these come from different runs or different averaging schemes, the paper should explain that clearly. As written, this looks like a stale table or scoring inconsistency.
- **ETS arbitrariness.** The composite ETS score is useful as a rough summary, but it feels somewhat arbitrary. The weights give Library Health the largest role, which matches the paper thesis, but the individual submetrics are more informative than the aggregate score. I would not want ETS to be treated as a definitive leaderboard metric without stronger justification.
- **Empirical evaluation is not as strong as the framing suggests.** Only Code Evol on Sonnet has multiple runs. The other main configurations appear to be single run evaluations. Since agent results can vary a lot across runs, this weakens the comparison. Some task categories and domains are also under sampled. API orchestration has only one session, and compose and regress tasks have only one task per session. This makes the domain level and composability claims tentative.
- **GPT 4o difficult to interpret.** All GPT 4o configurations produce the same low task completion and zero tools. The authors say this likely reflects a harness interaction rather than a model limitation. I appreciate the honesty, but this means the cross provider claims should be weakened.
- **Citation trust concerns.** Several references or claims appear inaccurate or possibly hallucinated. In particular, the paper's description of SkillsBench does not seem to match public descriptions of that work, and some cited titles or first authors for Tool Genesis, EvoSkill, and Vending Bench appear inconsistent with public records. This does not necessarily invalidate the benchmark, but it does lower confidence in the paper's scholarly positioning.

### Questions for the authors
1. Why do Table 2 and Table 3 report different ETS values for the Sonnet systems under the original weighting?
2. How exactly are Library Health and ETS computed when no new tools are created?
3. Are seed tools included in reuse, duplication, utilization, and safety calculations?
4. How is tool usage recorded for the reuse metric?
5. Why does One Shot create only 3 tools across 99 tasks while Code Evol creates around 21?
6. What caused GPT 4o to create zero tools across all systems?
7. Can the authors provide multi seed results for No Evolution, One Shot, and Strategy Only?
8. Will the full benchmark, hidden tests, adversarial inputs, task schemas, harness, and generated libraries be released?
9. Can the authors verify and correct the related work references and claims?

### Overall assessment
I like the direction of this paper, and I think the workshop audience would find the problem relevant. The benchmark idea is useful, and the reuse paradox is a good insight. However, the current version has too many issues for me to recommend acceptance confidently: underspecified metric edge cases, an apparent table inconsistency, limited multi seed evaluation, small sample sizes for some claims, difficult to interpret GPT 4o results, and citation problems. I would encourage the authors to revise and resubmit with clearer metric definitions, corrected tables, stronger reproducibility details, additional runs, and a cleaned up bibliography. In its current form, I rate it slightly below the acceptance threshold.

---

## Review 3 — Reviewer mKCT

**Title:** Review of AgentSkills '26 Paper #35
**Date:** 09 May 2026, 20:52 (modified 13 May 2026, 09:00)
**Rating:** **6: Marginally above acceptance threshold**
**Confidence:** 3 (fairly confident)

### Quality
- The paper addresses a timely and important problem: evaluating LLM-generated tool libraries beyond task completion.
- The use of hidden tests and adversarial inputs provides useful independent validation.
- Several configurations are based on single-run evaluations, so the reported results may be affected by run-to-run randomness in LLM outputs, tool creation, and reuse decisions. The empirical comparison would be stronger if all configurations were evaluated with multiple seeds.
- The tasks and domains appear to be author-designed for diagnostic purposes, and the paper does not clearly explain how representative they are of real-world development scenarios.
- The experimental coverage across models and systems is uneven. For example, Haiku is evaluated only for No-Evolution and Code-Evol, while Strategy-Only and One-Shot are not evaluated with Haiku. The paper does not explain this omission, which makes the cross-model comparison less complete.
- The regression setting may be too restrictive or under-specified. The paper does not clearly explain whether agents are allowed to modify previously generated tools. In real software development, requirement changes are a major reason why function inputs, outputs, and interfaces evolve. Therefore, simply checking whether previous behavior is preserved may conflate unintended regressions with legitimate interface changes caused by new requirements. The benchmark would be stronger if it explicitly modeled requirement evolution and distinguished expected API/interface changes from accidental regressions.

### Clarity
- The motivation is clear and easy to understand.
- The paper clearly explains why task completion alone is insufficient.
- The tables and figures help explain the evaluation setup and results.
- Equation (10) could be better presented.
- The ETS weighting choices need stronger justification.
- The paper should clarify why Strategy-Only and One-Shot are not evaluated on Haiku.

### Originality
- The paper is original in its evaluation perspective.
- It shifts the focus from task-level success to the quality of accumulated tool libraries.
- Applying software engineering concepts such as reuse, regression, duplication, and composability to LLM-generated tool libraries is interesting and useful.

### Significance
- The paper's claim is significant because LLM agents increasingly generate, store, and reuse tools over time.
- It shows that task completion alone can be misleading for evaluating self-evolving agents.
- The findings have practical implications for building safer and more reliable tool-generating agents.
- The significance is somewhat limited by the current experimental scale.
- The practical relevance would be stronger with more realistic long-running agent environments.
- Number and diversity could be enriched to enhance the significance of the dataset.

---

## Aggregate scores
| Reviewer | Rating | Confidence |
|----------|--------|------------|
| 5Wxk     | 6 (marginally above) | 4 |
| oR4e     | 5 (marginally below) | 4 |
| mKCT     | 6 (marginally above) | 3 |

Decision: **Reject** (workshop)

---

## What we're doing about it (for a friend picking this up)
The v2/kdd2026 revision under `paper/v2/` and `paper/kdd2026/` addresses these reviews:

- **W1 (LH predictive validity)** — added LH→TC correlation analysis (ρ=−0.03 across 144 sessions); reframed LH as a diagnostic, not a predictive signal.
- **W2 (representative baselines)** — added CREATOR-style, ToolMaker-style, ToolCoder-style implementations under `src/evolvetool_bench/baselines/`.
- **W3a (test independence)** — still author-written, documented honestly in Limitations.
- **W3b (test counts)** — hidden tests expanded from mean 1.9 → 5.1 per gap task; adversarial 2.7 → 5.8.
- **W3c (statistical power)** — multi-seed runs added for No-Evol, One-Shot, Strategy-Only, CREATOR-style on Sonnet (n=3–4 each).
- **ETS LH=0 stress test** — added to Table 2.
- **Empty-library edge cases** — metric behavior documented explicitly.
- **Table 2/3 inconsistency** — resolved.
- **Citation cleanup** — references.bib audited; added DreamCoder, Stitch, Babble, TOOLMAKER, EvoSkills.
- **Disclosure** — Code-Evol = ARISE; disclosed in the de-anonymized v2 preprint.
- **Reframing** — under the expanded test suite, per-tool correctness drops to 0–3% across all systems. The paper is now framed as a diagnostic-framework contribution, not a leaderboard.
