from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import queue
from tqdm import tqdm

from bsp.generator import SpeculativeGenerationModel

speculation_table = [
    (1, 7),
    (2, 6),
    (4, 4),
    (8, 4),
    (16, 3),
    (32, 1)
]

# speculation_table.reverse()

def get_speculate_len(batch_size, default=1):
    for upper_bound, spec_len in speculation_table:
        if batch_size <= upper_bound:
            return spec_len
    return default

@torch.inference_mode()
def generate_hf(prompts, model, tokenizer, step):
    tokenizer.padding_side='left'
    gen_conf = GenerationConfig(max_new_tokens=step, min_new_tokens=step, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    token_seqs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    token_seqs = token_seqs.to('cuda')
    out = model.generate(**token_seqs, generation_config=gen_conf)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

def server_func(args, q, finished_queue):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.fp16:
        model.half()
        model.cuda()

    assisted = args.assist_model is not None    
    if assisted:
        assist_model = AutoModelForCausalLM.from_pretrained(args.assist_model)
        if args.fp16:
            assist_model.half()
        assist_model.cuda()
        generator = SpeculativeGenerationModel(model, assist_model, tokenizer, args.speculate_step)

    finished = False
    cnt = 0
    finished_queue.put('ready')
    batch_size_record = []
    while not finished:
        # collect a batch
        batch = []
        batch_id = []
        while len(batch) < args.max_batch_size:
            try:
                prompt_id, prompt = q.get_nowait()
                if prompt_id < 0:
                    finished = True
                    break
                batch.append(prompt)
                batch_id.append(prompt_id)
            except queue.Empty:
                if len(batch) > 0: 
                    break
        if len(batch) == 0:
            break
        cnt += 1
        batch_size_record.append(len(batch))
        start_t = time.time()
        if assisted: 
            spec_len = args.speculate_step if args.speculate_step is not None else get_speculate_len(len(batch))
            generator.generate(batch, args.len_out, specualtive_step=spec_len)
        else:
            spec_len = 0
            generate_hf(batch, model, tokenizer, args.len_out)
        end_t = time.time()
        if args.print:
            print(f"[server], {end_t}, {end_t - start_t}, {len(batch)}, {spec_len}")
        for prompt_id in batch_id:
            finished_queue.put((prompt_id, time.time()))
    print("Average Batch size:", np.mean(batch_size_record))

def get_dataset(dataset_name, truncate = None, start_pos = None):
    from datasets import load_dataset
    if dataset_name == 'alespalla/chatbot_instruction_prompts':
        dataset = load_dataset(dataset_name)
        dataset = [t['prompt'] for t in dataset['test']]
        if start_pos is not None:
            dataset = dataset[start_pos:]
        if truncate is not None:
            return dataset[:truncate]
    else:
        raise ValueError("Unsupported dataset")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--assist-model', type=str)
    parser.add_argument('--tokenizer', type=str)
    parser.add_argument('--speculate-step', type=int)
    parser.add_argument('--len-out', type=int, default=128)
    parser.add_argument('--max-batch-size', type=int, default=32)
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num-samples', type=int)

    parser.add_argument('--interval', type=float)
    parser.add_argument('--cv', type=float)
    parser.add_argument('--dump-track', type=str)
    parser.add_argument('--load-track', type=str)
    args = parser.parse_args()
    print(args)
    start_t = time.time()

    q = mp.Queue()
    finished_queue = mp.Queue()
    server_proc = mp.Process(target=server_func, args=(args, q, finished_queue))
    server_proc.start()

    records = {}
    _ = finished_queue.get()
    if args.load_track is None:
        interval_record  = []
        num_samples = args.num_sampels
    else:
        interval_record = np.load(args.load_track)
        num_samples = interval_record.shape[0]
    prompts = get_dataset(args.dataset, num_samples, start_pos=200)
    for i, prompt in enumerate(prompts):
        records[i] = time.time()
        q.put((i, prompt))
        if args.load_track is None:
            k = 1 / (args.cv)**2
            theta = args.interval / k
            interval = np.random.gamma(k, scale=theta)
            interval_record.append(interval)
        else:
            interval = interval_record[i]
        time.sleep(interval)
    q.put((-1, ''))

    server_proc.join()
    latency_list = []
    t0 = records[0]
    print('t0', t0)
    while not finished_queue.empty():
        prompt_id, end_time = finished_queue.get()
        start_time = records[prompt_id]
        latency = end_time - start_time
        latency_list.append(latency)
        if args.print:
            print(f"[req {prompt_id}], {start_time - t0}, {latency}")
    print(f"Total time: {time.time() - start_t} s")
    print(f"Interval: average={np.mean(interval_record)} std={np.std(interval_record)}")
    print(f"Average latency: {np.mean(latency_list)} s")

    if args.dump_track is not None:
        np.save(args.dump_track, np.array(interval_record))