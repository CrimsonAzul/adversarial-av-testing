"""The imitation student with a classified lane decision (run d8 on).

d2-d7 used the PPO actor's mean action for both outputs: the lane command was regressed onto teacher labels
of -1/0/+1 and acted on above ±0.5. A least-squares fit averages the labels of similar states, so the
command hovered near the threshold. With weight 10 on lane-change samples, d6 was eager and wove across
lane lines; with weight 3, d7 was hesitant and clipped warning triangles (docs/LOG.md, 2026-11-03).

TacticalModel instead classifies the lane decision (left / keep / right, cross-entropy) and regresses
throttle/brake. Its lane command is P(left) - P(right), acted on above the controller's threshold, so the
tactical action has the same form as before and avtest.tactical's controller, the evaluator and
qualification are unchanged. Offline on d6's data it made 2.5% false lane changes per step against the
weighted fit's 14.9%, but caught 40% of the teacher's lane-change steps against 74% (LOG.md, 2026-11-03). It mimics the parts of the Stable-Baselines3 model interface this code uses (predict,
observation_space, save), and load_policy reads either kind of model file.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

LANE_VALUES = (1.0, 0.0, -1.0)          # class 0 left, 1 keep, 2 right (avtest.tactical LEFT, KEEP, RIGHT)
POLICY_ENTRY, META_ENTRY = "tactical_policy.pt", "tactical_meta.json"


def lane_classes(commands: np.ndarray) -> np.ndarray:
    """Teacher lane labels (-1/0/+1) as class indices."""
    commands = np.asarray(commands)
    return np.where(commands > 0.5, 0, np.where(commands < -0.5, 2, 1))


class TacticalNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: tuple = (256, 256)):
        super().__init__()
        layers, width = [], obs_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.Tanh()]
            width = size
        self.trunk = nn.Sequential(*layers)
        self.lane = nn.Linear(width, 3)
        self.throttle = nn.Linear(width, 1)

    def forward(self, x):
        z = self.trunk(x)
        return self.lane(z), torch.tanh(self.throttle(z)).squeeze(-1)


class TacticalModel:
    def __init__(self, obs_dim: int, hidden: tuple = (256, 256), seed: int = 0):
        import gymnasium as gym
        torch.manual_seed(seed)
        self.obs_dim, self.hidden = int(obs_dim), tuple(hidden)
        self.net = TacticalNet(self.obs_dim, self.hidden)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)

    def predict(self, obs, deterministic: bool = True):
        """(lane command, throttle). The lane command is P(left) - P(right): the controller's threshold
        (0.5) then changes lanes only when one side is the clear majority, and the threshold can be tuned
        after training (scripts.evaluate_controller) without retraining."""
        with torch.no_grad():
            logits, throttle = self.net(torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1))
            p = torch.softmax(logits, -1)[0]
        return np.array([float(p[0] - p[2]), float(throttle)], dtype=np.float32), None

    def save(self, path) -> None:
        buffer = io.BytesIO()
        torch.save(self.net.state_dict(), buffer)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(META_ENTRY, json.dumps({"obs_dim": self.obs_dim, "hidden": list(self.hidden)}))
            archive.writestr(POLICY_ENTRY, buffer.getvalue())

    @classmethod
    def load(cls, path) -> "TacticalModel":
        with zipfile.ZipFile(path) as archive:
            meta = json.loads(archive.read(META_ENTRY))
            state = torch.load(io.BytesIO(archive.read(POLICY_ENTRY)), map_location="cpu", weights_only=True)
        model = cls(meta["obs_dim"], tuple(meta["hidden"]))
        model.net.load_state_dict(state)
        return model


def is_tactical_model(path) -> bool:
    with zipfile.ZipFile(path) as archive:
        return POLICY_ENTRY in archive.namelist()


def load_policy(path):
    """A driver's policy: TacticalModel (d8 on) or the Stable-Baselines3 PPO model (d1-d7)."""
    if is_tactical_model(path):
        return TacticalModel.load(path)
    from stable_baselines3 import PPO
    return PPO.load(str(path), device="cpu")


def fit_classify(model: TacticalModel, obs: np.ndarray, act: np.ndarray, epochs: int, seed: int) -> dict:
    """Cross-entropy on the lane class plus mean-squared error on throttle/brake, over all data so far."""
    gen = torch.Generator().manual_seed(seed)
    classes = torch.as_tensor(lane_classes(act[:, 0]))
    throttle = torch.as_tensor(act[:, 1].astype(np.float32))
    optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    for _ in range(epochs):
        order = torch.randperm(len(act), generator=gen)
        total = 0.0
        for i in range(0, len(order), 512):
            idx = order[i:i + 512]
            logits, predicted = model.net(torch.as_tensor(obs[idx.numpy()].astype(np.float32)))
            loss = nn.functional.cross_entropy(logits, classes[idx]) + ((predicted - throttle[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
    return {"loss": total / len(order), **lane_metrics(model, obs[: min(len(obs), 20_000)], act[: min(len(act), 20_000)])}


def lane_metrics(model: TacticalModel, obs: np.ndarray, act: np.ndarray) -> dict:
    """Agreement with the teacher's lane class, recall of its lane changes, false changes, throttle error."""
    with torch.no_grad():
        logits, throttle = model.net(torch.as_tensor(obs.astype(np.float32)))
    chosen, truth = logits.argmax(-1).numpy(), lane_classes(act[:, 0])
    changes = truth != 1
    return {"lane_command_agreement": float(np.mean(chosen == truth)),
            "lane_change_recall": float(np.mean(chosen[changes] == truth[changes])) if changes.any() else None,
            "false_lane_changes": float(np.mean((chosen != 1) & ~changes)),
            "action_mae": float(np.mean(np.abs(np.stack([np.take(LANE_VALUES, chosen), throttle.numpy()], 1) - act)))}
