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
# Throttle/brake levels for a classified longitudinal action (d9 on), placed where the 60 km/h teacher's commands
# lie: 6.7% are hard braking near -1 (d8's data, 820k samples). d8 regressed throttle as one number, and 16 of
# its 18 reproduced collisions on the T1 dev panel came without braking (LOG.md, 2026-11-03).
THROTTLE_BINS = (-1.0, -0.7, -0.35, -0.12, 0.0, 0.12, 0.35, 0.7, 1.0)
POLICY_ENTRY, META_ENTRY = "tactical_policy.pt", "tactical_meta.json"


def throttle_classes(values: np.ndarray, bins=THROTTLE_BINS) -> np.ndarray:
    """Teacher throttle/brake labels as the index of the nearest level."""
    return np.abs(np.asarray(values)[:, None] - np.asarray(bins)[None, :]).argmin(1)


def lane_classes(commands: np.ndarray) -> np.ndarray:
    """Teacher lane labels (-1/0/+1) as class indices."""
    commands = np.asarray(commands)
    return np.where(commands > 0.5, 0, np.where(commands < -0.5, 2, 1))


class TacticalNet(nn.Module):
    """Lane logits, and throttle as a value in [-1, 1] (d8) or as logits over throttle_bins (d9 on)."""

    def __init__(self, obs_dim: int, hidden: tuple = (256, 256), throttle_bins: tuple | None = None):
        super().__init__()
        self.binned = throttle_bins is not None
        layers, width = [], obs_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.Tanh()]
            width = size
        self.trunk = nn.Sequential(*layers)
        self.lane = nn.Linear(width, 3)
        self.throttle = nn.Linear(width, len(throttle_bins) if self.binned else 1)

    def forward(self, x):
        z = self.trunk(x)
        throttle = self.throttle(z)
        return self.lane(z), throttle if self.binned else torch.tanh(throttle).squeeze(-1)


class TacticalModel:
    def __init__(self, obs_dim: int, hidden: tuple = (256, 256), seed: int = 0, throttle_bins: tuple | None = None):
        import gymnasium as gym
        torch.manual_seed(seed)
        self.obs_dim, self.hidden = int(obs_dim), tuple(hidden)
        self.throttle_bins = tuple(throttle_bins) if throttle_bins is not None else None
        self.net = TacticalNet(self.obs_dim, self.hidden, self.throttle_bins)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)

    def predict(self, obs, deterministic: bool = True):
        """(lane command, throttle). The lane command is P(left) - P(right): the controller's threshold
        (0.5) then changes lanes only when one side is the clear majority, and the threshold can be tuned
        after training (scripts.evaluate_controller) without retraining."""
        with torch.no_grad():
            logits, throttle = self.net(torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1))
            p = torch.softmax(logits, -1)[0]
        return np.array([float(p[0] - p[2]), self.throttle_value(throttle)[0]], dtype=np.float32), None

    def throttle_value(self, throttle) -> np.ndarray:
        """Throttle/brake commands from the network output: the most likely level, or the regressed value."""
        if self.throttle_bins is None:
            return throttle.reshape(-1).numpy()
        return np.asarray(self.throttle_bins, dtype=np.float32)[throttle.argmax(-1).numpy()]

    def save(self, path) -> None:
        buffer = io.BytesIO()
        torch.save(self.net.state_dict(), buffer)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(META_ENTRY, json.dumps({"obs_dim": self.obs_dim, "hidden": list(self.hidden),
                                                     "throttle_bins": self.throttle_bins}))
            archive.writestr(POLICY_ENTRY, buffer.getvalue())

    @classmethod
    def load(cls, path) -> "TacticalModel":
        with zipfile.ZipFile(path) as archive:
            meta = json.loads(archive.read(META_ENTRY))
            state = torch.load(io.BytesIO(archive.read(POLICY_ENTRY)), map_location="cpu", weights_only=True)
        model = cls(meta["obs_dim"], tuple(meta["hidden"]), throttle_bins=meta.get("throttle_bins"))
        model.net.load_state_dict(state)
        return model


def is_tactical_model(path) -> bool:
    with zipfile.ZipFile(path) as archive:
        return POLICY_ENTRY in archive.namelist()


# ---------------------------------------------------------------------------------------------------------------
# End-to-end driver (author's decision, LOG.md 2026-11-05): the network outputs MetaDrive's steering and throttle
# itself; no controller or rule acts on them. The rules (PID steering, emergency braking, the guards) stay in the
# teacher, which labels every state the student visits with its own final steering and throttle.
#
# Pilot 1 (LOG.md, 2026-11-01), the only earlier end-to-end student, weaved: an MLP on raw [0, 1] sensor values made
# steering errors (~0.04) as large as the corrections lane keeping needs (~0.02). This design addresses that with:
# - inputs standardised per feature (mean/std of the first round's data, stored in the model), so the small changes
#   in lane-line and heading values that steering depends on are not swamped by 720 lidar beams;
# - the change of every input over the last 0.2 s as extra inputs (FrameHistory): lateral drift, yaw and other
#   vehicles' motion, which one frame does not show (the scripted event vehicle reports no sideways velocity);
# - one steering output per lane intent (left / keep / right of the current lane, classified as before): a state in
#   which the teacher sometimes changes lane and sometimes not gets the steering of one or the other, not their
#   average, which would aim between two lanes (the conditional-imitation structure of Codevilla et al., 2018,
#   with the command predicted by the network itself);
# - steering loss in units of the labels' spread for each intent: lane keeping needs corrections of ~0.02 while a
#   lane change steers ~10x harder, and one common scale let the keeping errors count for almost nothing;
# - a learning rate annealed within each round.
E2E_POLICY_ENTRY, E2E_META_ENTRY = "end_to_end_policy.pt", "end_to_end_meta.json"
HISTORY_STEPS = 2                 # motion input: obs(t) - obs(t - 0.2 s); imitation stores every 2nd step
E2E_HIDDEN = (512, 512, 512)
INTENT_THRESHOLD = 0.3            # P(left) - P(right) beyond which the left / right steering output is used
INPUT_STD_FLOOR = 1e-3
INPUT_CLIP = 10.0
# The observation's first 64 values (avtest.sim, MetaDrive's LidarStateObservation): 24 road-edge beams, heading to the
# lane, speed, steering, last throttle and steering, yaw rate, 24 lane-line beams, and 10 navigation values. Then the
# 4 nearest vehicles (16), 720 lidar beams and, for end-to-end drivers, the lane model (avtest.sim.lane_model, 13).
# Given the lane intent, the teacher's steering is a PID toward a lane: a function of the ego's state and the lane
# geometry alone. ``steer_features`` gives the steering outputs their own network on those values (and their
# changes), so the 1,472 vehicle and lidar inputs, which steering does not depend on, cannot add error to it.
STATE_INPUTS = 64
STEER_HIDDEN = (256, 256)


def steering_features(obs_dim: int) -> tuple:
    """The ego-state values and, if the observation has it, the lane model (not the object list after it)."""
    from .sim import E2E_OBJ_OBS_DIM, E2E_OBS_DIM, OBS_DIM
    lane = tuple(range(OBS_DIM, E2E_OBS_DIM)) if obs_dim in (E2E_OBS_DIM, E2E_OBJ_OBS_DIM) else ()
    return tuple(range(STATE_INPUTS)) + lane


class FrameHistory:
    """The end-to-end network's input for one episode: the observation, and its change over the last HISTORY_STEPS
    steps (zero at the start). Observations are rounded to float16 first, as imitation data is stored, so the
    network sees identical values when training and when driving."""

    def __init__(self, steps: int = HISTORY_STEPS):
        self.steps = steps
        self.reset()

    def reset(self) -> None:
        self.frames: list[np.ndarray] = []

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float16).astype(np.float32).reshape(-1)
        self.frames.append(obs)
        if len(self.frames) > self.steps + 1:
            self.frames.pop(0)
        return np.concatenate([obs, obs - self.frames[0]])


def stored_features(obs: np.ndarray, first: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """FrameHistory's input for stored samples ``idx``: imitation keeps every HISTORY_STEPS-th step, so a sample's
    change is its difference to the previous stored sample of the same episode (zero at an episode's start)."""
    prev = np.where(first[idx], idx, idx - 1)
    x = obs[idx].astype(np.float32)
    return np.concatenate([x, x - obs[prev].astype(np.float32)], axis=1)


def mlp(width: int, hidden: tuple, out: int) -> nn.Sequential:
    layers = []
    for size in hidden:
        layers += [nn.Linear(width, size), nn.SiLU()]
        width = size
    return nn.Sequential(*layers, nn.Linear(width, out))


class EndToEndNet(nn.Module):
    """Standardised inputs, a SiLU trunk, and three heads: lane intent (3 logits: left, keep, right), one steering
    value per intent, and throttle/brake logits over THROTTLE_BINS. With ``steer_features`` (observation indices) the
    steering values come from a separate network on those observation values and their changes."""

    def __init__(self, in_dim: int, hidden: tuple, n_bins: int, steer_features: tuple | None = None,
                 steer_hidden: tuple = STEER_HIDDEN):
        super().__init__()
        layers, width = [], in_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.SiLU()]
            width = size
        self.trunk = nn.Sequential(*layers)
        self.intent = nn.Linear(width, 3)
        self.steer_features = tuple(steer_features) if steer_features else None
        if self.steer_features:
            idx = torch.as_tensor(self.steer_features, dtype=torch.long)
            self.register_buffer("steer_idx", torch.cat([idx, in_dim // 2 + idx]))
            self.steer_net = mlp(2 * len(idx), tuple(steer_hidden), 3)
        else:
            self.steer = nn.Linear(width, 3)
        self.throttle = nn.Linear(width, n_bins)
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))
        self.register_buffer("steer_scale", torch.ones(3))         # labels' spread per intent (left, keep, right)

    def forward(self, x):
        x = ((x - self.mean) / self.std).clamp(-INPUT_CLIP, INPUT_CLIP)
        z = self.trunk(x)
        steer = self.steer_net(x[:, self.steer_idx]) if self.steer_features else self.steer(z)
        return self.intent(z), steer * self.steer_scale, self.throttle(z)


class EndToEndModel:
    """MetaDrive's action (steering, throttle/brake) from the observation, with no controller. ``predict`` keeps the
    episode's frame history: call ``reset`` at the start of every episode (SUTDriver.start does)."""

    def __init__(self, obs_dim: int, hidden: tuple = E2E_HIDDEN, seed: int = 0, throttle_bins: tuple = THROTTLE_BINS,
                 intent_threshold: float = INTENT_THRESHOLD, steer_features: tuple | None = None,
                 steer_hidden: tuple = STEER_HIDDEN, steer_blend: bool = False, throttle_mean: bool = False):
        import gymnasium as gym
        torch.manual_seed(seed)
        self.obs_dim, self.hidden = int(obs_dim), tuple(hidden)
        self.throttle_bins, self.intent_threshold = tuple(throttle_bins), float(intent_threshold)
        # steer_blend: steering = the intent probabilities' mix of the three steering outputs, instead of the output of
        # the decided intent. d16's intent switched on 6.4% of steps, each switch a 0.42 steering jump on average:
        # 65% of all its steering change (LOG.md, 2026-11-06). Training is the same either way.
        self.steer_blend = bool(steer_blend)
        # throttle_mean: throttle = the probability-weighted mean of the throttle levels, instead of the most likely
        # level. With the most likely level, d17 braked hard where the teacher did not brake within 1 s on half of its
        # hard-brake steps on T3 teacher states (LOG.md, 2026-11-07): a 30% braking probability can win a step.
        self.throttle_mean = bool(throttle_mean)
        self.steer_features = tuple(steer_features) if steer_features else None
        self.steer_hidden = tuple(steer_hidden)
        self.net = EndToEndNet(2 * self.obs_dim, self.hidden, len(self.throttle_bins), self.steer_features,
                               self.steer_hidden)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)
        self.history = FrameHistory()
        self.last_intent = 0

    def reset(self) -> None:
        self.history.reset()

    def predict(self, obs, deterministic: bool = True):
        """(steering, throttle) for this step of the episode."""
        return self.act_features(self.history(obs)), None

    def intent_class(self, p) -> int:
        """0 left, 1 keep, 2 right: a side when P(left) - P(right) passes intent_threshold."""
        command = float(p[0] - p[2])
        return 0 if command > self.intent_threshold else 2 if command < -self.intent_threshold else 1

    def act_features(self, features) -> np.ndarray:
        with torch.no_grad():
            intent, steer, throttle = self.net(torch.as_tensor(np.asarray(features, dtype=np.float32)).reshape(1, -1))
            p = torch.softmax(intent, -1)[0].numpy()
        k = self.intent_class(p)
        self.last_intent = int(LANE_VALUES[k])
        outputs = np.clip(steer[0].numpy(), -1.0, 1.0)
        steering = float(p @ outputs) if self.steer_blend else float(outputs[k])
        if self.throttle_mean:
            pedal = float(torch.softmax(throttle, -1)[0].numpy() @ np.asarray(self.throttle_bins, dtype=np.float32))
        else:
            pedal = self.throttle_bins[int(throttle.argmax(-1)[0])]
        return np.array([steering, pedal], dtype=np.float32)

    def set_normalization(self, features: np.ndarray, steering: np.ndarray, intent: np.ndarray) -> None:
        """Input mean/std and the steering labels' spread per intent (-1/0/+1 labels; all labels' spread for an
        intent with fewer than 10 samples), from the first round's data; fixed afterwards."""
        classes = lane_classes(intent)
        spread = [np.std(steering[classes == k]) if np.sum(classes == k) >= 10 else np.std(steering) for k in range(3)]
        with torch.no_grad():
            self.net.mean.copy_(torch.as_tensor(features.mean(0), dtype=torch.float32))
            self.net.std.copy_(torch.as_tensor(np.maximum(features.std(0), INPUT_STD_FLOOR), dtype=torch.float32))
            self.net.steer_scale.copy_(torch.as_tensor(np.maximum(spread, 1e-3), dtype=torch.float32))

    def save(self, path) -> None:
        buffer = io.BytesIO()
        torch.save(self.net.state_dict(), buffer)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(E2E_META_ENTRY, json.dumps({"obs_dim": self.obs_dim, "hidden": list(self.hidden),
                                                         "throttle_bins": list(self.throttle_bins),
                                                         "intent_threshold": self.intent_threshold,
                                                         "history_steps": HISTORY_STEPS,
                                                         "steer_features": list(self.steer_features or ()),
                                                         "steer_blend": self.steer_blend,
                                                         "throttle_mean": self.throttle_mean,
                                                         "steer_hidden": list(self.steer_hidden)}))
            archive.writestr(E2E_POLICY_ENTRY, buffer.getvalue())

    @classmethod
    def load(cls, path) -> "EndToEndModel":
        with zipfile.ZipFile(path) as archive:
            meta = json.loads(archive.read(E2E_META_ENTRY))
            state = torch.load(io.BytesIO(archive.read(E2E_POLICY_ENTRY)), map_location="cpu", weights_only=True)
        if meta["history_steps"] != HISTORY_STEPS:
            raise ValueError(f"{path} was trained with a {meta['history_steps']}-step history")
        model = cls(meta["obs_dim"], tuple(meta["hidden"]), throttle_bins=tuple(meta["throttle_bins"]),
                    intent_threshold=meta["intent_threshold"], steer_features=tuple(meta.get("steer_features") or ()),
                    steer_hidden=tuple(meta.get("steer_hidden", STEER_HIDDEN)),
                    steer_blend=meta.get("steer_blend", False), throttle_mean=meta.get("throttle_mean", False))
        model.net.load_state_dict(state)
        return model


def is_end_to_end_model(path) -> bool:
    with zipfile.ZipFile(path) as archive:
        return E2E_POLICY_ENTRY in archive.namelist()


def fit_end_to_end(model: EndToEndModel, obs: np.ndarray, act: np.ndarray, first: np.ndarray, train: np.ndarray,
                   heldout: np.ndarray, epochs: int, seed: int, lr: float = 1e-3, lr_end: float = 5e-5,
                   batch: int = 512, brake_weight: float = 1.0) -> dict:
    """Fit on the stored samples ``train`` (indices): cross-entropy on the intent (act[:, 0] as -1/0/+1), squared
    error of the steering output of the teacher's intent in units of that intent's labels' spread (act[:, 1]), cross-entropy
    on the throttle level (act[:, 2]), in which the hard-braking levels (<= -0.5) weigh ``brake_weight``. The learning
    rate falls from lr to lr_end along a cosine over the round. Metrics on the samples ``heldout``, which are never
    trained on."""
    gen = torch.Generator().manual_seed(seed)
    intent = torch.as_tensor(lane_classes(act[:, 0]))
    steer = torch.as_tensor(act[:, 1].astype(np.float32))
    throttle = torch.as_tensor(throttle_classes(act[:, 2], model.throttle_bins))
    level_weight = torch.as_tensor([brake_weight if b <= -0.5 else 1.0 for b in model.throttle_bins], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=lr)
    steps = epochs * int(np.ceil(len(train) / batch))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1), eta_min=lr_end)
    scale = model.net.steer_scale.clone()
    model.net.train()
    total = 0.0
    for _ in range(epochs):
        order = torch.as_tensor(train)[torch.randperm(len(train), generator=gen)]
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x = torch.as_tensor(stored_features(obs, first, idx.numpy()))
            intent_logits, steer_out, throttle_logits = model.net(x)
            chosen = steer_out.gather(1, intent[idx][:, None])[:, 0]
            loss = (nn.functional.cross_entropy(intent_logits, intent[idx])
                    + (((chosen - steer[idx]) / scale[intent[idx]]) ** 2).mean()
                    + nn.functional.cross_entropy(throttle_logits, throttle[idx], weight=level_weight))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total += loss.item() * len(idx)
        total /= max(len(order), 1)
    model.net.eval()
    return {"loss": total, **end_to_end_metrics(model, obs, act, first, heldout)}


def end_to_end_metrics(model: EndToEndModel, obs: np.ndarray, act: np.ndarray, first: np.ndarray,
                       idx: np.ndarray) -> dict:
    """On samples ``idx``: the steering error of the output the driver would use (its own intent, so intent errors
    count) and of the teacher's intent's output (steering precision alone), on steps where the teacher keeps its lane
    and overall; the labels' spread there;
    intent agreement, lane-change recall and false changes; throttle error and hard-brake recall."""
    if len(idx) == 0:
        return {}
    with torch.no_grad():
        intent_logits, steer_out, throttle_logits = model.net(torch.as_tensor(stored_features(obs, first, idx)))
        p = torch.softmax(intent_logits, -1).numpy()
    command = p[:, 0] - p[:, 2]
    chosen = np.where(command > model.intent_threshold, 0, np.where(command < -model.intent_threshold, 2, 1))
    truth = lane_classes(act[idx, 0])
    steer_label, pedal_label = act[idx, 1], act[idx, 2]
    own = np.clip(steer_out.numpy()[np.arange(len(idx)), chosen], -1, 1)
    teacher_branch = np.clip(steer_out.numpy()[np.arange(len(idx)), truth], -1, 1)
    pedal = np.asarray(model.throttle_bins, dtype=np.float32)[throttle_logits.argmax(-1).numpy()]
    keep, changes, hard = truth == 1, truth != 1, pedal_label <= -0.5
    return {"heldout_samples": int(len(idx)),
            "steer_mae": float(np.mean(np.abs(own - steer_label))),
            "steer_mae_keep": float(np.mean(np.abs(own[keep] - steer_label[keep]))) if keep.any() else None,
            "steer_mae_teacher_intent": float(np.mean(np.abs(teacher_branch - steer_label))),
            "steer_mae_keep_teacher_intent": float(np.mean(np.abs(teacher_branch[keep] - steer_label[keep])))
            if keep.any() else None,
            "steer_label_std": float(np.std(steer_label)),
            "steer_label_std_keep": float(np.std(steer_label[keep])) if keep.any() else None,
            "lane_command_agreement": float(np.mean(chosen == truth)),
            "lane_change_recall": float(np.mean(chosen[changes] == truth[changes])) if changes.any() else None,
            "false_lane_changes": float(np.mean((chosen != 1) & keep)),
            "throttle_mae": float(np.mean(np.abs(pedal - pedal_label))),
            "hard_brake_recall": float(np.mean(pedal[hard] <= -0.5)) if hard.any() else None,
            "false_hard_brakes": float(np.mean(pedal[~hard] <= -0.5)) if (~hard).any() else None}


def load_policy(path):
    """A driver's policy: EndToEndModel (from d14), TacticalModel (d8-d13) or the Stable-Baselines3 PPO model
    (d1-d7)."""
    if is_end_to_end_model(path):
        return EndToEndModel.load(path)
    if is_tactical_model(path):
        return TacticalModel.load(path)
    from stable_baselines3 import PPO
    return PPO.load(str(path), device="cpu")


def fit_classify(model: TacticalModel, obs: np.ndarray, act: np.ndarray, epochs: int, seed: int) -> dict:
    """Cross-entropy on the lane class, plus cross-entropy on the throttle level (binned) or mean-squared
    error on the throttle value, over all data so far."""
    gen = torch.Generator().manual_seed(seed)
    classes = torch.as_tensor(lane_classes(act[:, 0]))
    if model.throttle_bins is not None:
        throttle = torch.as_tensor(throttle_classes(act[:, 1], model.throttle_bins))
    else:
        throttle = torch.as_tensor(act[:, 1].astype(np.float32))
    optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    for _ in range(epochs):
        order = torch.randperm(len(act), generator=gen)
        total = 0.0
        for i in range(0, len(order), 512):
            idx = order[i:i + 512]
            logits, predicted = model.net(torch.as_tensor(obs[idx.numpy()].astype(np.float32)))
            if model.throttle_bins is not None:
                throttle_loss = nn.functional.cross_entropy(predicted, throttle[idx])
            else:
                throttle_loss = ((predicted - throttle[idx]) ** 2).mean()
            loss = nn.functional.cross_entropy(logits, classes[idx]) + throttle_loss
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
    pedal = model.throttle_value(throttle)
    hard_brake = act[:, 1] <= -0.5
    return {"lane_command_agreement": float(np.mean(chosen == truth)),
            "lane_change_recall": float(np.mean(chosen[changes] == truth[changes])) if changes.any() else None,
            "false_lane_changes": float(np.mean((chosen != 1) & ~changes)),
            "throttle_mae": float(np.mean(np.abs(pedal - act[:, 1]))),
            "hard_brake_recall": float(np.mean(pedal[hard_brake] <= -0.5)) if hard_brake.any() else None,
            "action_mae": float(np.mean(np.abs(np.stack([np.take(LANE_VALUES, chosen), pedal], 1) - act)))}
