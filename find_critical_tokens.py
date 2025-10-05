from datasets import load_dataset

from critical_tokens.vllm_utils import *
from critical_tokens.prompts import *

import re
import logging
import argparse
import json
from tqdm import tqdm
from collections import defaultdict

def find_critical_tokens(model: LLM, prompt: str, system_prompt: str = None, threshold: float = 0.90, top_k: int = 5, rollout_n: int = 10) -> dict:
    # 1. Greedy generation
    logging.info("Running greedy decoding with token probabilities...")
    greedy_result = greedy_decoding_with_tokenprobs(model, prompt, system_prompt=system_prompt, top_k=top_k)
    logging.info("Greedy decoding complete.")
    logging.info(greedy_result["output"])

    # 2. Binary search
    logging.info("Running binary search for critical tokens...")
    head, tail = 0, len(greedy_result['token_ids']) - 1
    while head < tail:
        mid = (head + tail) // 2
        logging.info(f"Binary search iteration: head={head}, mid={mid}, tail={tail}")
        
        answer_dict = defaultdict(float)
        for alt_token in greedy_result['token_probs'][mid]:
            # obtain prefix with alt token
            partial_response_tokens = greedy_result['token_ids'][:mid] + [alt_token['token_id']]
            partial_response_str = model.get_tokenizer().decode(partial_response_tokens)
            # sample `rollout_n` responses
            sampling_result = sampling_from_middle(model, prompt, partial_response_tokens, system_prompt=system_prompt, n=rollout_n)["response_strs"]
            # Parse the answer between the last \boxed{}
            pattern = re.compile(r'\\boxed\{([^\}]*?)\}', re.DOTALL)
            answers = [
                pattern.findall(partial_response_str + response)[-1] if pattern.search(partial_response_str + response) else None
                for response in sampling_result
            ]
            for answer in answers:
                answer_dict[answer] += alt_token["probability"] / len(answers)
        logging.info(f"Answers and rollout probabilities: {answer_dict}")
        # Check if any answer exceeds the threshold
        critical = any(prob >= threshold for answer, prob in answer_dict.items() if answer is not None)
        if critical:
            tail = mid
        else:
            head = mid + 1
    # head -= 1 # head is the index we substitute top_k random tokens. Critical token is the one at `head-1` by definition
    return {
        "greedy_result": greedy_result,
        "critical_token_index": head,
        "critical_token": {
            "token_id": greedy_result['token_ids'][head],
            "text": greedy_result['token_probs'][head][0]['text'],
            "probability": greedy_result['token_probs'][head][0]['probability']
        },
        "majority_answer": max(answer_dict, key=answer_dict.get) if answer_dict else None
    }


def main(args):
    logging.info("Init VLLM...")
    model = LLM(
        model=args.model,
        trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization)
    logging.info("Complete!")

    logging.info("Load dataset...")
    dataset = load_dataset("jinulee-v/" + args.dataset)
    dataset = dataset["test"] if "test" in dataset else dataset["train"]
    # print stats
    logging.info(f"Dataset loaded with {len(dataset)} examples.")
    logging.info("Complete!")
    
    logging.info("\n\n-------------------------------------\n\n")

    results = []
    for i, example in enumerate(tqdm(dataset)):
        prompt = example["question"]
        system_prompt=SYSTEM_PROMPT[args.dataset]
        
        try:
            result = find_critical_tokens(model, prompt, system_prompt=system_prompt, threshold=args.critical_token_threshold) 
            results.append({
                "config": {
                    "model": args.model,
                    "dataset": "jinulee-v/" + args.dataset,
                    "top_k": args.top_k,
                    "rollout_n": args.rollout_n,
                    "critical_token_threshold": args.critical_token_threshold
                },
                "example": example,
                "critical_token": result
            })
        except Exception as e:
            print(e.__class__, e)
            pass
        model_alias = args.model.split("/")[-1]
        with open(f"data/critical_tokens_{args.dataset}_{model_alias}.jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLLM utilities for critical tokens.")
    
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for more verbose output")

    # vLLM args
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it", help="Model name to use with VLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6, help="GPU memory utilization for VLLM")
    # Dataset args
    parser.add_argument("--dataset", type=str, default="aime2024", choices=SYSTEM_PROMPT.keys(), help="Dataset name to load")
    # parser.add_argument("--split", type=str, default="aime2024", help="Dataset split to use")
    # Critical token args
    parser.add_argument("--top_k", type=int, default=5, help="Number of top tokens to consider for critical token analysis")
    parser.add_argument("--rollout_n", type=int, default=10, help="Number of rollouts for critical token analysis")
    parser.add_argument("--critical_token_threshold", type=float, default=0.90, help="Threshold for determining critical tokens")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        model = LLM(model="google/gemma-2-2b-it", trust_remote_code=True, gpu_memory_utilization=0.6, max_model_len=4096)
        print(find_critical_tokens(
            model=model,
            prompt=r"""Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:
    \[\log_2\left({x \over yz}\right) = {1 \over 2}\]
    \[\log_2\left({y \over xz}\right) = {1 \over 3}\]
    \[\log_2\left({z \over xy}\right) = {1 \over 4}\]
    Then the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.""",
            threshold=0.95,
            system_prompt=SYSTEM_PROMPT_MATH,
        ))
    else:
        main(args)
