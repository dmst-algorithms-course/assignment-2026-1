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
  
