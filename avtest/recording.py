"""Episode recordings: top-down GIF with a text overlay plus a keyframe strip (PNG).

Used to show progress: the same showcase scenarios are recorded after every driver
training round/chunk, and newly found failures are recorded during adversarial search
(the SUT failing, and a reference driver completing the same scene).

``record_episode`` runs one full episode, so it is also a valid evaluation of that driver
on that scenario; the returned summary has the same keys as avtest.sim.run_episode.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import scenario as sc

FRAME_EVERY = 2            # keep every 2nd env step (0.2 s); GIF plays at 2× real time
GIF_MS = 100
SIZE = 480
SCALING = 6                # pixels per metre: an 80 m view centred on the ego, heading up
KEYFRAMES = 6
try:
    FONT = ImageFont.truetype("arial.ttf", 13)
except OSError:                       # non-Windows systems without Arial
    FONT = ImageFont.load_default()


def showcase(tier: str, count_per_event: int = 1, panel_seed: int = 77) -> list[sc.Scenario]:
    """Fixed scenarios, one per event type of the tier, on validation-split seeds."""
    from . import seeds
    rng = np.random.default_rng(panel_seed)
    out = []
    events = sc.TIERS[tier]["events"]
    map_seeds = seeds.panel_seeds("val", count_per_event * len(events), panel_seed)
    for i, event in enumerate(events):
        for k in range(count_per_event):
            theta = sc.with_event(rng.random(sc.DIM), event, tier)
            out.append(sc.decode(theta, map_seeds[i * count_per_event + k], tier))
    return out


def _frame(env) -> Image.Image:
    raw = env.render(mode="topdown", window=False, screen_size=(SIZE, SIZE), scaling=SCALING,
                     target_agent_heading_up=True, num_stack=1)
    return Image.fromarray(np.asarray(raw)[:, :, :3].astype(np.uint8))        # MetaDrive returns (h, w, rgb)


def _overlay(img: Image.Image, lines: list[str], colour=(255, 255, 255)) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    height = 16 * len(lines) + 8
    draw.rectangle([0, 0, img.width, height], fill=(20, 20, 20))
    for i, line in enumerate(lines):
        draw.text((6, 4 + 16 * i), line, fill=colour, font=FONT)
    return img


def record_episode(env, scenario: sc.Scenario, driver, path: Path, title: str) -> dict:
    """Drive one episode with ``driver`` and write <path>.gif and <path>_keyframes.png."""
    import time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env.load(scenario)
    obs, info = env.reset()
    driver.start(env)
    v = scenario.values
    header = f"{title} | {driver.name}"
    detail = (f"{v['map']} lanes {v['lane_num']} traffic {v['traffic_density']:.2f} | {v['event']} "
              f"{v['event_lane']} {v['event_ahead_m']:.0f} m, dv {v['event_rel_speed_mps']:+.1f} m/s, "
              f"t {v['event_trigger_s']:.1f} s")
    frames, steps, reward, started = [], 0, 0.0, time.time()
    while True:
        obs, r, terminated, truncated, info = env.step(driver.act(env, obs))
        reward += r
        steps += 1
        if steps % FRAME_EVERY == 0 or terminated or truncated:
            status = info["outcome"].upper() if (terminated or truncated) else "driving"
            frames.append(_overlay(_frame(env), [header, detail,
                                                 f"t {steps * 0.1:5.1f} s  speed {env.agent.speed * 3.6:4.0f} km/h  {status}"]))
        if terminated or truncated:
            break
    final = info["outcome"]
    colour = (120, 255, 120) if final == "success" else (255, 110, 110)
    frames.extend([_overlay(frames[-1], [header, detail, f"END: {final.upper()} after {steps * 0.1:.1f} s"], colour)] * 8)
    frames[0].save(path.with_suffix(".gif"), save_all=True, append_images=frames[1:], duration=GIF_MS, loop=0,
                   optimize=False)
    picks = np.linspace(0, len(frames) - 9, KEYFRAMES).astype(int).tolist() + [len(frames) - 1]
    strip = Image.new("RGB", (SIZE * len(picks) // 2, SIZE // 2), "white")
    for i, j in enumerate(picks):
        strip.paste(frames[j].resize((SIZE // 2, SIZE // 2)), (i * SIZE // 2, 0))
    strip.save(path.parent / (path.name + "_keyframes.png"))
    return {"driver": driver.name, "outcome": final, "steps": steps, "reward": reward,
            "route_completion": float(info.get("route_completion", 0.0)), **env.telemetry,
            "event_triggered_step": env.engine.managers["event_manager"].triggered_at,
            "wall_s": time.time() - started, "gif": str(path.with_suffix(".gif"))}
