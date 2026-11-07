"""PPO fine-tuning of an end-to-end driver (avtest.policy.EndToEndModel) from its imitation weights (from r1;
LOG.md, 2026-11-07).

Why: three imitation runs (d16-d18) reach 87-89% on the T3 dev panel against their teacher's 95.6%. The teacher's
emergency and predictive brakes are thresholds on the current state, so near a threshold its labels say "brake" or
"go" for nearly identical inputs, and the network learns ~50% there and brakes late or releases early. Reinforcement
learning scores the network's own braking by what happens next (collision or not) instead of by agreement with the
teacher, and can therefore also do better than the teacher.

What is trained: the network's throttle/brake output, as a policy over its THROTTLE_BINS levels (a categorical
distribution from the throttle logits). Lane intent and steering are decided as when driving (EndToEndModel
act_features) and are not explored; they share the network's trunk with the throttle, so distillation losses keep
their outputs at the imitation network's (``ref``) while the trunk changes. A KL penalty keeps the throttle policy
near the imitation policy (KL(policy || ref), which lets it settle on one of two levels the imitation network found
equally likely, as at a braking threshold). The driver is still the network alone: when it drives, the most likely
level is used, as before.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .policy import INPUT_CLIP, LANE_VALUES, HISTORY_STEPS, EndToEndModel, mlp

VALUE_HIDDEN = (512, 512)
REWARD_SCALE = 0.1          # the critic learns returns in tenths of the environment's reward (success +20, failure -30)


class ValueNet(nn.Module):
    """The critic: the expected discounted return from the policy's own input (standardised as the policy's)."""

    def __init__(self, in_dim: int, hidden: tuple = VALUE_HIDDEN):
        super().__init__()
        self.body = mlp(in_dim, tuple(hidden), 1)
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))

    @classmethod
    def for_model(cls, model: EndToEndModel, hidden: tuple = VALUE_HIDDEN) -> "ValueNet":
        critic = cls(2 * model.obs_dim, hidden)
        with torch.no_grad():
            critic.mean.copy_(model.net.mean)
            critic.std.copy_(model.net.std)
        return critic

    def forward(self, x):
        return self.body(((x - self.mean) / self.std).clamp(-INPUT_CLIP, INPUT_CLIP))[:, 0]


def sample_action(model: EndToEndModel, features, rng: np.random.Generator) -> tuple[np.ndarray, int, float]:
    """(action, throttle level index, its log-probability): the steering exactly as EndToEndModel.act_features
    gives it, the throttle level drawn from the throttle policy."""
    with torch.no_grad():
        intent, steer, throttle = model.net(torch.as_tensor(np.asarray(features, dtype=np.float32)).reshape(1, -1))
        p = torch.softmax(intent, -1)[0].numpy()
        logp = torch.log_softmax(throttle.double(), -1)[0].numpy()
    k = model.intent_class(p)
    model.last_intent = int(LANE_VALUES[k])
    outputs = np.clip(steer[0].numpy(), -1.0, 1.0)
    steering = float(p @ outputs) if model.steer_blend else float(outputs[k])
    probs = np.exp(logp)
    level = int(rng.choice(len(probs), p=probs / probs.sum()))
    return np.array([steering, model.throttle_bins[level]], dtype=np.float32), level, float(logp[level])


def episode_features(obs16: np.ndarray, steps: int = HISTORY_STEPS) -> np.ndarray:
    """FrameHistory's inputs for every step of one episode from its float16 observations: each observation and its
    change over the last ``steps`` steps (to the first observation while fewer have passed)."""
    x = np.asarray(obs16, dtype=np.float16).astype(np.float32)
    prev = x[np.maximum(np.arange(len(x)) - steps, 0)]
    return np.concatenate([x, x - prev], axis=1)


def gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """(advantages, returns) of one episode that ended by termination (no bootstrap after its last step)."""
    adv = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        following = values[t + 1] if t + 1 < len(rewards) else 0.0
        delta = rewards[t] + gamma * following - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
    return adv.astype(np.float32), (adv + values).astype(np.float32)


def batched(fn, x: torch.Tensor, size: int = 8192):
    """fn over x in chunks (no gradient): tuples of outputs are concatenated per element."""
    with torch.no_grad():
        parts = [fn(x[i:i + size]) for i in range(0, len(x), size)]
    if isinstance(parts[0], tuple):
        return tuple(torch.cat([p[k] for p in parts]) for k in range(len(parts[0])))
    return torch.cat(parts)


def ppo_update(model: EndToEndModel, ref: EndToEndModel, features: torch.Tensor, levels: torch.Tensor,
               old_logp: torch.Tensor, advantages: torch.Tensor, optimizer, epochs: int = 4, batch: int = 1024,
               clip: float = 0.2, kl_coef: float = 0.05, distill_coef: float = 10.0, target_kl: float = 0.02,
               seed: int = 0) -> dict:
    """Clipped PPO on the throttle policy, plus kl_coef * KL(policy || ref) on the throttle levels and distillation
    of the intent (KL(ref || policy)) and steering outputs (squared error in units of each intent's label spread) to
    ``ref``. Advantages are standardised over the batch. Epochs stop early once the policy has moved more than
    1.5 x target_kl (approximate KL from the collecting policy)."""
    gen = torch.Generator().manual_seed(seed)
    ref_intent, ref_steer, ref_throttle = batched(ref.net, features)
    ref_logp_all = torch.log_softmax(ref_throttle, -1)
    ref_intent_logp = torch.log_softmax(ref_intent, -1)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    scale = model.net.steer_scale.detach()
    model.net.train()
    stats, stopped = [], False
    for epoch in range(epochs):
        order = torch.randperm(len(features), generator=gen)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            intent, steer, throttle = model.net(features[idx])
            logp_all = torch.log_softmax(throttle, -1)
            logp = logp_all.gather(1, levels[idx][:, None])[:, 0]
            log_ratio = logp - old_logp[idx]
            ratio = torch.exp(log_ratio)
            surrogate = torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx])
            p_all = torch.exp(logp_all)
            kl_ref = (p_all * (logp_all - ref_logp_all[idx])).sum(-1).mean()
            intent_logp = torch.log_softmax(intent, -1)
            distill_intent = (torch.exp(ref_intent_logp[idx]) * (ref_intent_logp[idx] - intent_logp)).sum(-1).mean()
            distill_steer = (((steer - ref_steer[idx]) / scale) ** 2).mean()
            loss = -surrogate.mean() + kl_coef * kl_ref + distill_coef * (distill_intent + distill_steer)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                approx_kl = float(((ratio - 1) - log_ratio).mean())
                stats.append({"policy_loss": float(-surrogate.mean()), "kl_ref": float(kl_ref),
                              "distill_intent": float(distill_intent), "distill_steer": float(distill_steer),
                              "approx_kl": approx_kl, "clip_fraction": float(((ratio - 1).abs() > clip).float().mean()),
                              "entropy": float(-(p_all * logp_all).sum(-1).mean())})
            if approx_kl > 1.5 * target_kl:
                stopped = True
                break
        if stopped:
            break
    model.net.eval()
    out = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    out.update({"epochs_run": epoch + 1, "stopped_early": stopped})
    return out


def critic_update(critic: ValueNet, features: torch.Tensor, returns: torch.Tensor, optimizer, epochs: int = 4,
                  batch: int = 1024, seed: int = 0) -> dict:
    """Squared error of the critic on the batch's returns (in REWARD_SCALE units)."""
    gen = torch.Generator().manual_seed(seed + 1)
    critic.train()
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(features), generator=gen)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            loss = ((critic(features[idx]) - returns[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    critic.eval()
    return {"value_loss": float(np.mean(losses))}


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    var = float(np.var(returns))
    return float(1 - np.var(returns - values) / var) if var > 0 else 0.0
