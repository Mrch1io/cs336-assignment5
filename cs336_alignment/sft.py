import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ALPACA_SFT_PROMPT = (Path(__file__).parent / "prompts_safety" / "alpaca_sft.prompt").read_text().strip()


class PackedSFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        with open(dataset_path) as f:
            docs = [json.loads(line) for line in f]
        if shuffle:
            random.shuffle(docs)
        tokens = []
        for doc in docs:
            text = ALPACA_SFT_PROMPT.format(instruction=doc["prompt"], response=doc["response"])
            tokens += tokenizer.encode(text) + [tokenizer.eos_token_id]
        n = (len(tokens) - 1) // seq_length
        self.input_ids = torch.tensor(tokens[: n * seq_length]).reshape(n, seq_length)
        self.labels = torch.tensor(tokens[1 : n * seq_length + 1]).reshape(n, seq_length)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def parse_mmlu_response(mmlu_example, model_output):
    match = re.search(r"The correct answer is ([A-D])", model_output)
    return match.group(1) if match else None


def parse_gsm8k_response(model_output):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", model_output)
    return numbers[-1] if numbers else None


def sequence_log_prob(model, input_ids):
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    return log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).sum()


def compute_per_instance_dpo_loss(lm, lm_ref, tokenizer, beta, prompt, response_chosen, response_rejected):
    chosen = tokenizer.encode(ALPACA_SFT_PROMPT.format(instruction=prompt, response=response_chosen))
    rejected = tokenizer.encode(ALPACA_SFT_PROMPT.format(instruction=prompt, response=response_rejected))
    chosen = torch.tensor([chosen + [tokenizer.eos_token_id]])
    rejected = torch.tensor([rejected + [tokenizer.eos_token_id]])
    with torch.no_grad():
        ref_chosen = sequence_log_prob(lm_ref, chosen.to(lm_ref.device))
        ref_rejected = sequence_log_prob(lm_ref, rejected.to(lm_ref.device))
    policy_chosen = sequence_log_prob(lm, chosen.to(lm.device))
    policy_rejected = sequence_log_prob(lm, rejected.to(lm.device))
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected).to(policy_chosen.device)
    return -F.logsigmoid(beta * logits)
