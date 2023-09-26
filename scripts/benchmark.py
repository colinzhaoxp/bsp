from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
# import deepspeed
# from optimum.bettertransformer import BetterTransformer

from tqdm import tqdm

from bsp.generator import SpeculativeGenerationModel

@torch.inference_mode()
def generate_hf(prompts, model, tokenizer, step):
    tokenizer.padding_side='left'
    gen_conf = GenerationConfig(max_new_tokens=step, min_new_tokens=step, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    token_seqs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    token_seqs = token_seqs.to('cuda')
    model = model.to('cuda')
    out = model.generate(**token_seqs, generation_config=gen_conf)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

@torch.inference_mode()
def generate_hf_assist(prompts, model, assist_model, tokenizer, step):
    tokenizer.padding_side='left'
    gen_conf = GenerationConfig(max_new_tokens=step, min_new_tokens=step, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    ret = []
    for p in prompts:
        token_seqs = tokenizer(p, return_tensors="pt")
        token_seqs = token_seqs.to('cuda')
        model = model.to('cuda')
        out = model.generate(**token_seqs, generation_config=gen_conf, assistant_model=assist_model)
        ret.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return ret

def get_dataset(dataset_name, truncate = None):
    from datasets import load_dataset
    if dataset_name == 'alespalla/chatbot_instruction_prompts':
        dataset = load_dataset(dataset_name)
        dataset = [t['prompt'] for t in dataset['test']]
        if truncate is not None:
            return dataset[:truncate]
    else:
        raise ValueError("Unsupported dataset")

def benchmark(gen_fn, prompts, batch_size, warmup=3):
    for _ in range(warmup):
        out = gen_fn(prompts[:batch_size])
    data_loader = DataLoader(prompts, batch_size=batch_size, shuffle=True)
    generated_seqs = []
    torch.cuda.synchronize()
    start_t = time.time()
    for prompt in tqdm(data_loader):
        generated_seqs.extend(gen_fn(prompt))
    torch.cuda.synchronize()
    dur = time.time() - start_t
    return dur, generated_seqs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--assist-model', type=str)
    parser.add_argument('--tokenizer', type=str)
    parser.add_argument('--speculate-steps', type=int, nargs='+')
    parser.add_argument('--len-out', type=int)
    parser.add_argument('--batch-sizes', type=int, nargs='+')
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--collect-stats', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset-truncate', type=int)
    args = parser.parse_args()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    # model = deepspeed.init_inference(model, replace_with_kernel_inject=True)
    # model = BetterTransformer.transform(model)

    assist_model = AutoModelForCausalLM.from_pretrained(args.assist_model)
    prompts = get_dataset(args.dataset, args.dataset_truncate)
    if args.fp16:
        model.half()
        assist_model.half()
    model.cuda()
    assist_model.cuda()
    print("batch size, speculate step, sec/token")
    for batch_size in args.batch_sizes:
        for speculate_step in args.speculate_steps:
            assist_model.max_assistant_tokens = speculate_step
            generator = SpeculativeGenerationModel(model, assist_model, tokenizer, speculate_step)
            if speculate_step == 0:
                t, ret = benchmark(lambda p: generate_hf(p, model, tokenizer, args.len_out), prompts, batch_size)
            else:
                t, ret = benchmark(lambda p: generator.generate(p, args.len_out, collect_stats=args.collect_stats), prompts, batch_size)
            num_tokens = len(ret) * args.len_out
            print(f"{batch_size}, {speculate_step}, {t / num_tokens}")
            
            if args.collect_stats:
                hit_rate, time_speculate, time_verify, verify_calls = generator.get_stats()
                print("speculation hit rate:", ', '.join([str(h.cpu().numpy()) for h in hit_rate]))
                print("expected correct speculated length:", hit_rate.sum())
                print(f"time for speculation {time_speculate} s | verification {time_verify} s | #verifys: {verify_calls}")