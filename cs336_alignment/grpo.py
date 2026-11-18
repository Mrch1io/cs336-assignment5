import torch
import torch.nn.functional as F


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    max_len = max(len(p) + len(o) for p, o in zip(prompt_ids, output_ids))
    input_ids = torch.full((len(prompt_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
    response_mask = torch.zeros((len(prompt_ids), max_len), dtype=torch.bool)
    for i, (p, o) in enumerate(zip(prompt_ids, output_ids)):
        input_ids[i, : len(p) + len(o)] = torch.tensor(p + o)
        response_mask[i, len(p) : len(p) + len(o)] = True
    return {"input_ids": input_ids[:, :-1], "labels": input_ids[:, 1:], "response_mask": response_mask[:, 1:]}


def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    result = {"log_probs": log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)}
    if return_token_entropy:
        result["token_entropy"] = -(log_probs.exp() * log_probs).sum(dim=-1)
    return result


def compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths):
    rewards = [reward_fn(r, g) for r, g in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([r["reward"] for r in rewards], dtype=torch.float32)
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": sum(r["format_reward"] for r in rewards) / len(rewards),
        "mean_answer_reward": sum(r["answer_reward"] for r in rewards) / len(rewards),
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards, group_size, baseline="mean", advantage_eps=1e-6, advantage_normalizer="std"
):
    grouped = raw_rewards.reshape(-1, group_size)
    advantages = grouped - grouped.mean(dim=1, keepdim=True) if baseline == "mean" else grouped
    if advantage_normalizer == "std":
        advantages = advantages / (grouped.std(dim=1, keepdim=True) + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = advantages / (grouped.mean(dim=1, keepdim=True) + advantage_eps)
    return advantages.reshape(-1), {"mean_reward": raw_rewards.mean().item()}


def compute_policy_gradient_loss(
    raw_rewards_or_advantages,
    policy_log_probs,
    importance_reweighting_method="none",
    old_log_probs=None,
    cliprange=None,
    response_mask=None,
):
    advantages = raw_rewards_or_advantages.reshape(-1, 1)
    if importance_reweighting_method == "none":
        return -advantages * policy_log_probs, {}
    log_ratio = policy_log_probs - old_log_probs
    if importance_reweighting_method == "gspo":
        log_ratio = (log_ratio * response_mask).sum(dim=1, keepdim=True) / response_mask.sum(dim=1, keepdim=True)
    ratio = torch.exp(log_ratio).expand_as(policy_log_probs)
    if importance_reweighting_method == "noclip":
        return -ratio * advantages, {}
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages)
    return loss, {"clip_fraction": (ratio != clipped_ratio).float().mean()}


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss, mask, loss_normalization="sequence", normalization_constant=None
):
    masked = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        return (masked.sum(dim=1) / mask.sum(dim=1)).mean()
    return masked.sum() / normalization_constant


def grpo_train_step(
    model,
    tokenizer,
    optimizer,
    gradient_accumulation_steps,
    max_grad_norm,
    reward_fn,
    repeated_prompts,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    baseline="mean",
    advantage_eps=1e-6,
    advantage_normalizer="std",
    importance_reweighting_method="none",
    old_log_probs=None,
    cliprange=None,
    loss_normalization="sequence",
    normalization_constant=None,
):
    raw_rewards, metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, _ = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    device = next(model.parameters()).device
    microbatch_size = len(rollout_responses) // gradient_accumulation_steps
    total_loss = torch.zeros((), device=device)
    for i in range(0, len(rollout_responses), microbatch_size):
        input_ids = batch["input_ids"][i : i + microbatch_size].to(device)
        labels = batch["labels"][i : i + microbatch_size].to(device)
        mask = batch["response_mask"][i : i + microbatch_size].to(device)
        old = None if old_log_probs is None else old_log_probs[i : i + microbatch_size].to(device)
        log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            advantages[i : i + microbatch_size].to(device),
            log_probs,
            importance_reweighting_method,
            old,
            cliprange,
            mask,
        )
        loss = aggregate_loss_across_microbatch(per_token_loss, mask, loss_normalization, normalization_constant)
        if loss_normalization == "sequence":
            loss = loss / gradient_accumulation_steps
        loss.backward()
        total_loss += loss.detach()
    if max_grad_norm is not None:
        metadata["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    metadata.update(loss_metadata)
    return total_loss, metadata
