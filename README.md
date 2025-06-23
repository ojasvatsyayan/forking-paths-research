


This project analyzes where large language models generate divergent completions ("forking paths") when prompted with multiple-choice questions from the MMLU benchmark.

## Objective

To identify the index where two completions from Qwen diverge. This index will be classified as a forking token, and will be vital to next steps of our research.

## Setup

- Model: Qwen1.5-1.8B (via Hugging Face Transformers)
- Dataset: MMLU (High School Government and Politics subset)
- Environment: Python + Jupyter Notebooks

## Method

1. Each question from the MMLU subset is prompted twice to Qwen.
2. Two completions are generated per question.
3. The earliest divergence in tokens between the two completions is recorded.
4. Output is saved in both JSON and CSV formats.

## Files

- `testingforforks.ipynb`: Code for model rollout and fork detection
- `forking_results.json`: Full results in structured format
- `forking_results.csv`: Flattened version for analysis
=======
# Forking Paths in Language Model Rollouts

This research project explores *forking paths* — points of divergence in language model generations — by analyzing multiple completions from the same prompt. We investigate when and how two rollouts deviate, what alternative tokens cause divergence, and whether different paths still lead to correct answers.

##  Experiments

### 1. MMLU Prompt Divergence (`mmlu_prompt_fork.ipynb`)
We used the **MMLU dataset** (`high_school_government_and_politics`) to:
- Format multiple-choice questions into prompts
- Generate two completions per prompt using different random seeds
- Identify **forking index** where the completions begin to diverge
- Log completions, forking points, and correctness
- Save results to `forking_results.csv` and `forking_results.json`

*Goal:* Understand whether different paths still arrive at the correct answer — and how early divergence impacts outcome.

---

### 2. Math Prompt Fork Sampling (`math_prompt_fork.ipynb`)
We tested forking behavior on a fixed math prompt:

> "What is 2 + 3? Think step by step and enclose your final answer in \boxed{}."

Steps:
- Generate deterministic completions
- Analyze token-level alternatives at each step (`top_k` sampling)
- Resample completions conditioned on **prefix + alt token**
- Evaluate which paths still return the correct answer (`\boxed{5}`)

Results saved to `fork_analysis.csv`

*Goal:* Explore whether high-likelihood but non-greedy continuations still lead to the right answer.

---

## Model Used
- `Qwen/Qwen1.5-1.8B` from Hugging Face
- Loaded with `transformers` and run with FP16 on GPU

---

## File Overview
| File | Description |
|------|-------------|
| `mmlu_prompt_fork.ipynb` | MMLU experiment notebook |
| `math_prompt_fork.ipynb` | Math prompt experiment |
| `forking_results.csv` | MMLU completions |
| `forking_results.json` | MMLU completions (JSON) |
| `fork_analysis.csv` | Math completions summary |

---



