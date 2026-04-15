import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, cache_utils, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
import time
import datasets
from datasets import load_dataset
import json


from origin import origin_generate, origin_warmup, sampling_generate
from tree_decoding import tree_generate, tree_warmup
from modification_test import tree_generate as modified_tree_generate
from run import run_bench_mark
from task import Task, HumanEvalTask, CNNSumTask
from model_type import ModelType
from transformers import logging
from run import Metric
from typing import List
import os
from model_io import load_tokenizer, load_causal_lm

logging.set_verbosity_error()

import sys
import argparse

os.environ['HF_HOME'] = '/pscratch/sd/h/hmuki/.cache/huggingface'
sys.setrecursionlimit(3000)



def run_task(model_type, model, tokenizer ,task: Task, data_num: range, tree_params, origin_params):
    ds = task.get_ds()
    warmup_prompt = "This is a test"
    warmup_num_beams = 3
    warmup_max_tokens = 200
    warmup_eos_token_ids = [model.config.eos_token_id]


    path = f"out/{model_type.name}/sample/{task.type().name}"
    os.makedirs(path, exist_ok=True)
    print("processing sample " )
    # with open(f"{path}/sample.jsonl", "w") as out_file:
    #     metrics = run_bench_mark(model, tokenizer, ds.select(data_num), sampling_generate, task, model_type, None, 1000)
    #     for metric in metrics:
    #         out_file.write(json.dumps(metric.to_dict()) + "\n")

    tree_warmup(model, tokenizer, warmup_prompt, warmup_num_beams, warmup_max_tokens, warmup_eos_token_ids)

    for parameter in tree_params:
        if parameter[0] == 1:
            continue

        path = f"out/{model_type.name}/tree/{task.type().name}"
        os.makedirs(path, exist_ok=True)
        print("processing tree ",parameter[0], "_",parameter[1] )
        with open(f"{path}/{parameter[0]}_{parameter[1]}.jsonl", "w") as out_file:
            metrics = run_bench_mark(model, tokenizer, ds.select(data_num), tree_generate, task, model_type, parameter[0], parameter[1])
            for metric in metrics:
                out_file.write(json.dumps(metric.to_dict()) + "\n")

    origin_warmup(model, tokenizer, warmup_prompt, warmup_num_beams, warmup_max_tokens)


    for parameter in origin_params:
        path = f"out/{model_type.name}/origin/{task.type().name}"
        os.makedirs(path, exist_ok=True)
        print("processing origin ",parameter[0], "_",parameter[1] )
        with open(f"{path}/{parameter[0]}_{parameter[1]}.jsonl", "w") as out_file:
            metrics = run_bench_mark(model, tokenizer, ds.select(data_num), origin_generate, task, model_type, parameter[0], parameter[1])
            for metric in metrics:
                out_file.write(json.dumps(metric.to_dict()) + "\n")


def name(type):
    match type:
        case ModelType.LLAMA3:
            return  "meta-llama/Llama-3.1-8B-Instruct"
        case ModelType.PHI35:
            return "microsoft/Phi-3.5-mini-instruct"
        case ModelType.MISTRAL:
            return "/pscratch/sd/h/hmuki/models/Mistral-Small-24B-Instruct-2501"
        case ModelType.LLAMA3_70B:
            return "meta-llama/Llama-3.1-70B-Instruct"
        case ModelType.REASONING:
            return "openai/gpt-oss-20b"


def test_model(model_type:ModelType, task: Task, tree_params, origin_params, data_num: range):
    print(model_type, tree_params)
    tokenizer = load_tokenizer(name(model_type))
    model = load_causal_lm(
        name(model_type),
        device_map="auto",
        torch_dtype=torch.float16
    )

    run_task(model_type, model, tokenizer, task, data_num, tree_params, origin_params)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-start", type=int, default=0)
    parser.add_argument("--data-num", type=int, default=100)
    parser.add_argument(
        "--task",
        choices=["human_eval", "cnn"],
        default="human_eval",
        help="Benchmark task to run.",
    )
    parser.add_argument(
        "--beam-widths",
        type=str,
        nargs="+",
        default=["3"],
        help="One or more beam widths to evaluate, e.g. --beam-widths 3 5 or --beam-widths 3,5.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum number of new tokens to generate for each run.",
    )
    return parser.parse_args()


def parse_beam_widths(values: List[str]) -> List[int]:
    beam_widths: List[int] = []
    for value in values:
        cleaned = value.strip().strip("[]")
        for part in cleaned.split(","):
            part = part.strip()
            if part:
                beam_widths.append(int(part))
    return beam_widths


def build_parameters(beam_widths: List[int], max_tokens: int) -> List[tuple[int, int]]:
    return [(beam_width, max_tokens) for beam_width in beam_widths]


def build_task(task_name: str) -> Task:
    match task_name:
        case "cnn":
            return CNNSumTask()
        case _:
            return HumanEvalTask()

if __name__ == "__main__":
    args = parse_args()
    data_num = range(args.data_start, args.data_start + args.data_num)
    task = build_task(args.task)

    beam_widths = parse_beam_widths(args.beam_widths)
    parameters = build_parameters(beam_widths, args.max_tokens)
    trie_paramters = build_parameters(beam_widths, args.max_tokens)

    #test_model(ModelType.PHI35, task, trie_paramters, parameters, data_num)
    #test_model(ModelType.LLAMA3, task, trie_paramters, parameters, data_num)
    #test_model(ModelType.MISTRAL, task, trie_paramters, parameters, data_num)
    #test_model(ModelType.PHI35, [(3, 400)], [], data_num)
    #from modification_test import write_out
    #write_out()
    #test_model(ModelType.LLAMA3, task, [(3,1000), (9,1000), (15,1000)], [], data_num)
    test_model(ModelType.MISTRAL, task, trie_paramters, parameters, data_num)
