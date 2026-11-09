# Reading list: misaligned sub-agents, steering, and conductor contamination

Compiled 2026-10-01 for the AAMAS project in `initial-ideas.md`. Every arXiv ID below was checked against the arXiv API or the paper's page.

Tiers: **[MUST]** read closely before designing experiments · **[CORE]** read the method and results sections · **[SKIM]** abstract and figures are enough, plus a citation.

---

## 1. Closest prior work (read first, since this decides what you can claim is new)

| Paper | Why it matters for you |
|---|---|
| **[MUST]** Luo et al., *AgentLens: Interpretable Safety Steering via Mechanistic Subspaces for Multi-Turn Coding Agent* (Jun 2026) [2606.22673](https://arxiv.org/abs/2606.22673) | The paper your abstract positions against. It does white-box detection and steering on multi-turn **coding agents** (Llama-3.1-8B, Qwen-2.5-7B, Gemma-2-9B) with a 10-dim subspace at one layer. The controller is external to a single agent. Your novelty is a corrector that takes part in the conversation, plus bidirectional influence. Check its subspace method and its benchmark (194 tasks). |
| **[MUST]** Herbster et al., *Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence* (Apr 2026) [2604.08169](https://arxiv.org/abs/2604.08169) | Induces misalignment with **malicious system prompts** (your arm 1) on dishonesty, then compares fixed-coefficient steering with **projection-aware steering** gated by a logistic-regression boundary (StTP and StMP). That is close to your "probe-guided steering tool". It also covers coherence loss directly. Uses Qwen3.6-27B. |
| **[MUST]** Bergen et al. (Goodfire), *Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations* (Sep 2026) [2609.19101](https://arxiv.org/abs/2609.19101) · [blog](https://www.goodfire.com/research/reward-hacking-activation-monitors) | **Difference-of-means probes for reward hacking on SWE-bench and DeepSWE**, compared against LLM monitors. This is the state of the art for your detection component, and it is single-agent. Copy their evaluation format (detection at matched FPR against an LLM-judge baseline). |
| **[CORE]** Jaipersaud, Krueger, Lubana, *How Do LLMs Persuade? Linear Probes Can Uncover Persuasion Dynamics in Multi-Turn Conversations* (Aug 2025) [2508.05625](https://arxiv.org/abs/2508.05625) | The closest precedent for **per-turn probe readings of persuasion between two LLM agents**. Borrow how they set up persuader and persuadee and how they locate the turns where persuasion happens. |
| **[CORE]** Wilhelm et al., *Monitoring Emergent Reward Hacking During Generation via Internal Activations* (Mar 2026) [2603.04069](https://arxiv.org/abs/2603.04069) | Token-level reward-hacking detection during generation on **fine-tuned adapters**, which matches your GRPO arm. Reports that the signal appears early in generation. |
| **[CORE]** Dutta, *Linear Probes add little for Verifiable Reward Hacking* (LessWrong, Jul 2026) [link](https://www.greaterwrong.com/posts/NzzmNREX4qR54q33j/linear-probes-add-little-for-verifiable-reward-hacking) | A negative result with **GRPO on a buggy reward** (Qwen2.5-0.5B). Probes reached 0.998 AUC, but string matching on the output did just as well. Reviewers will ask the same about your setup: if test integrity can be scored automatically, why use probes? You need an answer ready (see §6). |

## 2. The phenomenon: reward hacking on coding tasks, and how it generalizes into misalignment

These motivate the GRPO arm and give you a realistic environment.

- **[MUST]** MacDiarmid et al. (Anthropic/Redwood), *Natural Emergent Misalignment from Reward Hacking in Production RL* (Nov 2025) [2511.18397](https://arxiv.org/abs/2511.18397). Reward hacking learned in coding RL environments generalizes to sabotage and alignment faking. Your strongest motivation for "sub-agents RL'd on narrow objectives". Also covers **inoculation prompting**, a mitigation you may want as a baseline.
- **[MUST]** Zhong, Raghunathan, Carlini, *ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases* (Oct 2025, ICLR 2026) [2510.20270](https://arxiv.org/abs/2510.20270). Coding tasks where the spec and the tests contradict each other, so **any pass is cheating**. This gives you automatic test-integrity scoring out of the box. Consider using it instead of building your own environment.
- **[CORE]** Baker et al. (OpenAI), *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation* (Mar 2025) [2503.11926](https://arxiv.org/abs/2503.11926). Reward hacking on real coding tasks (e.g. editing tests to make them pass), monitored through chain-of-thought. Training against the monitor produced obfuscation.
- **[CORE]** Betley et al., *Emergent Misalignment* (Feb 2025) [2502.17424](https://arxiv.org/abs/2502.17424). Narrow fine-tuning produced broad misalignment. Expect reviewers to bring this up for the GRPO arm.
- **[SKIM]** *Unintended Misalignment from Agentic Fine-Tuning* (Aug 2025) [2508.14031](https://arxiv.org/abs/2508.14031). Agentic fine-tuning by itself erodes alignment.

## 3. Steering and reading: the method papers you need (in this order)

- **[MUST]** Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition* (ACL 2024) [2312.06681](https://arxiv.org/abs/2312.06681). This is CAA. Read the dataset construction (A/B answer-letter pairs), the layer sweep, and the **sycophancy** results.
- **[MUST]** Chen et al. (Anthropic), *Persona Vectors: Monitoring and Controlling Character Traits in Language Models* (Jul 2025) [2507.21509](https://arxiv.org/abs/2507.21509). **The single most useful template for your pipeline:** automated extraction of contrast pairs from a trait description, difference of means over response tokens, **monitoring shifts by projection** (including shifts caused by fine-tuning), steering, and coherence and MMLU checks. Your "corner-cutting direction" is a persona vector.
- **[MUST]** Im & Li, *A Unified Understanding and Evaluation of Steering Methods* (Feb 2025) [2502.02716](https://arxiv.org/abs/2502.02716). Theory and experiments showing **diff-of-means is the right default** over PCA and classifier directions. Cite it to justify your method choice.
- **[MUST]** Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024) [2406.11717](https://arxiv.org/abs/2406.11717). Diff-of-means plus **directional ablation** (projecting the direction out). The cleanest exposition of the extraction recipe, layer and position selection, and the add-vs-ablate distinction.
- **[CORE]** Zou et al., *Representation Engineering* (2023) [2310.01405](https://arxiv.org/abs/2310.01405). PCA reading vectors (LAT), and honesty and lie detection. Also the source of the read/control framing.
- **[CORE]** Marks & Tegmark, *The Geometry of Truth* (2023) [2310.06824](https://arxiv.org/abs/2310.06824). Mass-mean (diff-of-means) probes are **more causally relevant** than logistic-regression probes. This is why you read with a probe but write with diff-of-means.
- **[CORE]** Lee et al., *Programming Refusal with Conditional Activation Steering (CAST)* (2024) [2409.05907](https://arxiv.org/abs/2409.05907). Steering that runs only when a condition-vector projection fires. Formal prior art for your "probe-gated steering tool".
- **[CORE]** Wu et al., *AxBench* (Jan 2025) [2501.17148](https://arxiv.org/abs/2501.17148). **Prompting beats steering**, and diff-of-means beats SAEs. You must include a strong prompting baseline, which your "argue in text" condition already partly covers.
- **[CORE]** Tan et al., *Analyzing the Generalization and Reliability of Steering Vectors* (2024) [2407.12404](https://arxiv.org/abs/2407.12404), plus Braun et al., *Understanding (Un)Reliability of Steering Vectors* [2505.22637](https://arxiv.org/abs/2505.22637) and [2602.17881](https://arxiv.org/abs/2602.17881). Steering effects vary widely across inputs. The predictor of reliability is how well the classes **separate along the DiM direction**. Measure this before you run anything.
- **[SKIM]** Turner et al., *ActAdd* [2308.10248](https://arxiv.org/abs/2308.10248). The original single-pair activation addition. Li et al., *Inference-Time Intervention* [2306.03341](https://arxiv.org/abs/2306.03341). Probe-selected attention-head steering for truthfulness.
- **[SKIM]** *Analysing the Safety Pitfalls of Steering Vectors* (Mar 2026) [2603.24543](https://arxiv.org/abs/2603.24543). CAA vectors can raise jailbreak success by up to 57%. Relevant if you steer the conductor or apply the steering tool to the sub-agent.

## 4. Do directions survive fine-tuning? (needed for the GRPO arm)

- **[MUST]** Soligo, Turner et al., *Convergent Linear Representations of Emergent Misalignment* [2506.11618](https://arxiv.org/abs/2506.11618) and *Model Organisms for Emergent Misalignment* [2506.11613](https://arxiv.org/abs/2506.11613). Misalignment caused by fine-tuning shows up as a shared linear direction, which can be ablated. Small open models make good model organisms.
- **[CORE]** Wang et al. (OpenAI), *Persona Features Control Emergent Misalignment* [2506.19823](https://arxiv.org/abs/2506.19823). A "misaligned persona" feature predicts and controls misalignment caused by fine-tuning.
- **[CORE]** Persona Vectors (above), §5–6: projection shifts predict which fine-tunes will change the trait.

## 5. Probe evasion and deception (for the threat model and the limitations section)

- **[CORE]** Goldowsky-Dill et al. (Apollo), *Detecting Strategic Deception Using Linear Probes* [2502.03407](https://arxiv.org/abs/2502.03407). A realistic deception-probe evaluation: train on simple contrast data, then test on agentic deception.
- **[CORE]** Hubinger et al., *Sleeper Agents* [2401.05566](https://arxiv.org/abs/2401.05566), and the follow-up blog post, *Simple probes can catch sleeper agents* (Anthropic, 2024). Generic contrast-pair diff-of-means probes detect defection.
- **[CORE]** Gupta & Jenner, *RL-Obfuscation* [2506.14261](https://arxiv.org/abs/2506.14261), and Bailey et al., *Obfuscated Activations Bypass Latent-Space Defenses* [2412.09565](https://arxiv.org/abs/2412.09565). RL can learn to evade token-level probes. Max-pooled probes hold up better. **Keep the probe out of the GRPO reward**, and say so.
- **[SKIM]** Chen et al., *Reasoning Models Don't Always Say What They Think* [2505.05410](https://arxiv.org/abs/2505.05410). Background for "verbal agreement vs internal change".

## 6. Multi-agent side (your AAMAS framing)

- **[MUST]** Hammond et al., *Multi-Agent Risks from Advanced AI* (Cooperative AI Foundation, 2025) [2502.14143](https://arxiv.org/abs/2502.14143). The framing paper for this community. Map your threat onto its taxonomy (miscoordination, conflict, collusion, and the risk factors).
- **[CORE]** Huang et al., *On the Resilience of LLM-Based Multi-Agent Collaboration with Faulty Agents* [2408.00989](https://arxiv.org/abs/2408.00989). Injects faulty agents into hierarchical, flat, and other topologies, and finds **hierarchical structures most resilient**. This is the behavioral baseline for your conductor topology.
- **[CORE]** *Who's the Mole? … Intention-Hiding Malicious Agents* [2507.04724](https://arxiv.org/abs/2507.04724). Covert task-sabotaging agents in centralized and layered multi-agent systems, detected from **behavior only**. You can present your work as the white-box counterpart.
- **[CORE]** Papadopoulos, …, Lindsey, *Mind Viruses: Self-Propagating Ideas in Multi-Agent LLM Systems* (Aug 2026) [2608.10218](https://arxiv.org/abs/2608.10218). Behavioral contagion between coding agents. A system-prompt warning gave almost full immunity, so **"warn the conductor" is a cheap baseline you should include**.
- **[CORE]** Sharma et al., *Towards Understanding Sycophancy* [2310.13548](https://arxiv.org/abs/2310.13548). Conductor contamination is close to sycophancy. CAA also has a sycophancy vector you could use as a control direction.
- **[SKIM]** Kong et al., *Multi-LLM Systems Exhibit Robust Semantic Collapse* [2605.17193](https://arxiv.org/abs/2605.17193). In multi-agent convergence, steering out a sycophancy direction **did not** stop the collapse. A useful caution for how you interpret contamination.
- **[SKIM]** Khan et al., *Debating with More Persuasive LLMs Leads to More Truthful Answers* [2402.06782](https://arxiv.org/abs/2402.06782). Background on persuasion through debate.
- **[SKIM]** Lee & Tiwari, *Prompt Infection* [2410.07283](https://arxiv.org/abs/2410.07283). Ramesh & Li, *Communicating Activations Between LM Agents* [2501.14082](https://arxiv.org/abs/2501.14082). Cite as related work and move on.

---

## 7. Which steering-vector computations are in scope

Your vector does three jobs. Keep them separate in the paper, because the best method differs for each:

1. **Read**: project activations to score corner-cutting per turn, for both agents.
2. **Write (induce)**: arm 2 of the misalignment design, steering the sub-agent toward corner-cutting.
3. **Write (correct)**: the conductor's white-box tool.

### In scope (recommended)

| Method | What it is | Use it for | Notes |
|---|---|---|---|
| **Diff-of-means / CAA** (your default) | `v = mean(h⁺) − mean(h⁻)` at layer *L*. CAA is the version with paired prompts. | Reading and writing | Theoretically and empirically the best default (Im & Li; AxBench; Bergen et al.). Decide **(a) paired vs unpaired**, **(b) token position**: the A/B answer token as in CAA, or the mean over response tokens as in Persona Vectors (better for free-form agent output), and **(c) layer**: sweep it, pick by separation plus steering effect, usually around the middle layers. |
| **PCA of paired differences** (RepE/LAT) | First PC of `{h⁺ᵢ − h⁻ᵢ}` | Ablation only | Cheap robustness check. Report cosine similarity with DiM. Expect it to be similar or worse. |
| **Logistic-regression probe** | The weight vector *w* of a linear classifier | **Reading and gating only** | Usually better at classification, worse as a steering direction (Marks & Tegmark). Use it for detection thresholds and AUROC, and DiM for interventions. |
| **Directional ablation** | `h ← h − (h·v̂)v̂` | Correction | Removes the corner-cutting component without pushing toward "honest". Causes less coherence loss than additive steering. A natural second correction operator next to additive steering. |
| **Projection clamping / conditional (CAST, StTP-style)** | Intervene only when `h·v̂ > τ`, setting the projection to a target value | **The probe-guided steering tool** | This is what "white-box tool guided by a probe reading" should mean concretely. Cite CAST and Herbster et al. |
| **Additive with coefficient α** | `h ← h + α v` | Arm 2 (inducing) | Sweep α. Report the dose-response curve and coherence loss at each α. |

### Optional (only if DiM doesn't steer multi-turn agent behavior well)

- **BiPO**, a steering vector optimized with a preference objective ([2406.00045](https://arxiv.org/abs/2406.00045)). Fixes the case where DiM separates the classes but doesn't change behavior. Costs one small optimization per vector.

### Out of scope (cite, don't run)

- **SAE-feature steering**: needs SAEs for your exact model, AxBench shows it underperforms DiM, and it adds a whole axis to the study.
- **ReFT/LoRA-style learned interventions**: these are fine-tuning, which blurs the distinction with your GRPO arm.
- **Steering vector fields, flow-matching steering, MELBO (unsupervised)**: too new or too off-topic, and they add little to your question.
- **LEACE / concept erasure** ([2306.03819](https://arxiv.org/abs/2306.03819)): mention at most as the principled form of ablation.
- **Training-time "preventative steering"** (Persona Vectors §6): interesting, but it changes the research question.

---

## 8. Design pitfalls the literature flags for *this* experiment

1. **Circularity in arm 2.** If you induce misalignment with vector *v* and then detect it with *v*, detection is trivially perfect. Either extract the detection vector from **separate contrast data**, or report arm-2 detection as a sanity check rather than a result.
2. **Content vs disposition confound in conductor contamination.** The conductor's projection onto "corner-cutting" will go up just from *reading* text about skipping tests. Measure on the conductor's **own generated tokens** (or the last prompt token before it responds, as in Persona Vectors). Validate against a **behavioral** contamination outcome: does the conductor then approve bad code, or cut corners on its next task? Add a control where the sub-agent *describes* corner-cutting neutrally.
3. **Direction transfer to the GRPO model.** Re-extract *v* on the GRPO checkpoint and report the cosine with the base-model *v* (Soligo et al.; Persona Vectors). If they diverge, the bidirectional measure is comparing projections onto different axes.
4. **"Why not just check the tests?"** Dutta (2026) shows probes add little when hacks can be verified. Your answer is that the contribution is the *internal persuasion dynamics* (does verbal agreement match internal change, and is the conductor pulled toward the sub-agent), not detection alone. Say so explicitly.
5. **Baselines reviewers will expect.** A strong prompting baseline (AxBench), a "warn the conductor in its system prompt" baseline (Mind Viruses), and an LLM-judge monitor (Bergen et al.).
6. **Report steerability before steering.** Measure class separation along *v* (Braun et al.). If separation is poor, steering will be unreliable whatever α you choose.
7. **Coherence metrics.** Follow Persona Vectors (MMLU drop plus an LLM-judged coherence score) or AxBench (concept, instruction, and fluency scores) so the cost of correction is comparable with prior work.
