import argparse
import math

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--n-ctx", dest="n_ctx", type=int, default=2048)
    parser.add_argument(
        "--begin-context-tokens",
        dest="begin_context_tokens",
        type=int,
        default=512
    )

    parser.add_argument("input_file")
    parser.add_argument("out_file")
  
        args = parser.parse_args()

    if args.stride <= 0:
        parser.error("--stride πρέπει να είναι θετικός ακέραιος")

    if args.n_ctx <= 1:
        parser.error("--n-ctx πρέπει να είναι μεγαλύτερο από 1")

    if args.begin_context_tokens <= 0:
        parser.error("--begin-context-tokens πρέπει να είναι θετικός ακέραιος")

    if args.begin_context_tokens + args.stride > args.n_ctx:
        parser.error(
            "--begin-context-tokens + --stride δεν πρέπει να ξεπερνά το --n-ctx"
        )
    return args
def stable_log_probs(row):
    max_val = max(row)
    shifted = [x - max_val for x in row]
    log_sum_exp = math.log(sum(math.exp(x) for x in shifted))
    return [x - log_sum_exp for x in shifted]
