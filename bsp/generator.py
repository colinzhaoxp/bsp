from typing import List
import torch
import time

def _run_and_timing(fn):
    torch.cuda.synchronize()
    start_t = time.time()
    ret = fn()
    torch.cuda.synchronize()
    dur = time.time() - start_t
    return ret, dur

class SpeculativeGenerationModel:
    def __init__(self, model, assist_model, tokenizer, specualtive_step=1, device='cuda'):
        self.model = model.to(device)
        self.assist_model = assist_model.to(device)
        self.tokenizer = tokenizer
        self.device=device

        self.specualtive_step = 1 if specualtive_step is None else specualtive_step

        # stats
        self.pos_correct = torch.zeros([self.specualtive_step], device=device)
        self.pos_cnt = 0

        self.time_speculate = 0
        self.time_verify = 0
        self.verify_calls = 0

    def _speculative(self, input_ids, attention_mask, kv_cache, speculate_step):
        batch_size = input_ids.shape[0]
        generated_tokens = [[] for _ in range(batch_size)]
        for i in range(speculate_step):
            ret = self.assist_model(input_ids,
                                    attention_mask=attention_mask, 
                                    use_cache=True, 
                                    past_key_values=kv_cache)
            input_ids = torch.argmax(ret.logits[:, -1:], axis=2)

            for b in range(batch_size):
                generated_tokens[b].append(input_ids[b, 0])

            attention_mask = self._extend_mask(attention_mask) 
            kv_cache = ret.past_key_values
        return generated_tokens, attention_mask, kv_cache
    
    def _last_pos_logits(self, logits, mask):
        last_pos = torch.sum(mask, axis=1) - 1
        return logits[torch.arange(logits.shape[0]), last_pos]
    
    def _extend_mask(self, mask):
        return torch.cat([mask, torch.ones([mask.shape[0], 1], device=self.device, dtype=torch.int32)], axis=1)

    @torch.inference_mode()
    def generate(self, prompts:List[str], num_out:int, collect_stats=False, specualtive_step=None):
        specualtive_step = self.specualtive_step if specualtive_step is None else specualtive_step
        self.tokenizer.padding_side='right'
        token_seqs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        batch_size = len(prompts)
        assist_kv_cache = None
        input_ids = token_seqs['input_ids'].to(self.device)
        attention_mask = input_attention_mask = token_seqs['attention_mask'].to(self.device)
        prompt_len = attention_mask.sum(axis=1)

        # prefill
        ret, t_prefill = _run_and_timing(lambda: self.model(input_ids, attention_mask=input_attention_mask, use_cache=True))
        self.time_verify += t_prefill
        self.verify_calls += 1
        first_token = torch.argmax(self._last_pos_logits(ret.logits, attention_mask), axis=1).unsqueeze(1) 
        attention_mask = self._extend_mask(attention_mask)
        input_ids = torch.cat([input_ids, first_token], axis=1)
        kv_cache = ret.past_key_values
        generated_tokens = input_ids
        valid_lens = torch.ones(batch_size, device=self.device) 

        # stats
        while True:
            (speculated_tokens, attention_mask, assist_kv_cache), t_spec = _run_and_timing(lambda: self._speculative(input_ids, attention_mask, assist_kv_cache, specualtive_step))
            self.time_speculate += t_spec
            # verify
            speculated_tokens = torch.tensor(speculated_tokens, device=self.device, dtype=torch.int64)
            verify_inputs = torch.cat([first_token, speculated_tokens], axis=1)
            ret, t_verify = _run_and_timing(lambda: self.model(verify_inputs, attention_mask=attention_mask, use_cache=True, past_key_values=kv_cache))
            self.time_verify += t_verify
            self.verify_calls += 1
            logits = ret.logits
            kv_cache = ret.past_key_values
            correct = logits[:, :-1].argmax(dim=2)

            # mask wrong predictions
            check_mask = torch.cumsum(correct == speculated_tokens, 1) == torch.arange(1, specualtive_step + 1, device=self.device)

            correct_len = torch.sum(check_mask, axis=1)
            first_token = torch.argmax(logits[torch.arange(logits.shape[0]), correct_len], axis=1).unsqueeze(1)
            input_ids = torch.concat([speculated_tokens[:, -1:], first_token], axis=1)
            attention_mask[:, -specualtive_step:] = check_mask
            attention_mask = self._extend_mask(attention_mask)
            generated_tokens = torch.cat([generated_tokens, speculated_tokens, first_token], axis=1)

            # update stats
            if collect_stats: 
                not_ended = (valid_lens < num_out).unsqueeze(1)
                self.pos_correct += (check_mask * not_ended).sum(dim=0)
                self.pos_cnt += not_ended.sum() 

            valid_lens += correct_len + 1
            if torch.all(valid_lens >= num_out):
                break
        ret = []
        for b in range(batch_size):
            valid_token = torch.nonzero(attention_mask[b], as_tuple=True)[0]
            tokens = generated_tokens[b][valid_token][:prompt_len[b] + num_out]
            ret.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
        
        return ret

    def get_stats(self):
        return self.pos_correct / self.pos_cnt, self.time_speculate, self.time_verify, self.verify_calls

    def reset_stats(self):
        self.pos_correct = 0
        self.pos_cnt = 0
        self.time_speculate = 0
        self.time_verify = 0
        self.verify_calls = 0