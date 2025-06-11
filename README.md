

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
