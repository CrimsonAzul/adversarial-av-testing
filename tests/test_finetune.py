"""PPO fine-tuning of end-to-end drivers (avtest.finetune), on synthetic data."""
import numpy as np
import torch

from avtest.finetune import (ValueNet, batched, critic_update, episode_features, explained_variance, gae, ppo_update,
                             sample_action)
from avtest.policy import EndToEndModel, FrameHistory


def small_model(seed: int = 0) -> EndToEndModel:
    """As the real drivers: steering from its own network on the leading inputs, intent and throttle from the trunk."""
    return EndToEndModel(8, (32, 32), seed=seed, steer_features=(0, 1, 2, 3), steer_hidden=(16,))


def test_episode_features_equal_the_frame_history_when_driving():
    rng = np.random.default_rng(0)
    obs = rng.random((7, 8)).astype(np.float32)
    history = FrameHistory()
    driving = np.stack([history(o) for o in obs])
    assert np.array_equal(episode_features(obs.astype(np.float16)), driving)


def test_sampled_action_steers_as_when_driving_and_draws_levels_by_probability():
    model = small_model()
    features = np.random.default_rng(1).random(16).astype(np.float32)
    driven = model.act_features(features)
    rng = np.random.default_rng(2)
    draws = [sample_action(model, features, rng) for _ in range(4000)]
    assert all(a[0] == driven[0] for a, _, _ in draws)                        # steering exactly as driven
    assert all(a[1] == model.throttle_bins[level] for a, level, _ in draws)
    with torch.no_grad():
        logp = torch.log_softmax(model.net(torch.as_tensor(features)[None])[2], -1)[0].numpy()
    freq = np.bincount([level for _, level, _ in draws], minlength=len(logp)) / len(draws)
    assert np.abs(freq - np.exp(logp)).max() < 0.03
    assert all(abs(lp - logp[level]) < 1e-5 for _, level, lp in draws)


def test_gae_with_lambda_one_gives_discounted_returns():
    rewards = np.array([1.0, 0.0, 2.0, -3.0])
    values = np.array([0.5, -0.2, 0.1, 0.4])
    adv, ret = gae(rewards, values, gamma=0.9, lam=1.0)
    expected = [sum(0.9 ** k * r for k, r in enumerate(rewards[t:])) for t in range(4)]
    assert np.allclose(ret, expected, atol=1e-5) and np.allclose(adv, ret - values, atol=1e-5)
    adv0, _ = gae(rewards, values, gamma=0.9, lam=0.0)                       # one-step TD errors
    assert np.allclose(adv0, rewards + 0.9 * np.append(values[1:], 0.0) - values, atol=1e-5)


def synthetic_batch(model: EndToEndModel, n: int, seed: int):
    """States with x0 > 0.5 reward full braking (level 0) and punish everything else; other states the reverse."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, 16)).astype(np.float32)
    x[:, 8:] = 0.0
    features = torch.as_tensor(x)
    with torch.no_grad():
        logp_all = torch.log_softmax(model.net(features)[2], -1)
    levels = torch.multinomial(torch.exp(logp_all), 1, generator=torch.Generator().manual_seed(seed))[:, 0]
    old_logp = logp_all.gather(1, levels[:, None])[:, 0]
    brake_state = x[:, 0] > 0.5
    good = (levels.numpy() == 0) == brake_state
    return features, levels, old_logp, torch.as_tensor(np.where(good, 1.0, -1.0).astype(np.float32)), brake_state


def brake_probability(model, features, mask):
    with torch.no_grad():
        p = torch.softmax(model.net(features)[2], -1)[:, 0].numpy()
    return float(p[mask].mean()), float(p[~mask].mean())


def test_ppo_update_learns_where_to_brake_and_keeps_intent_and_steering():
    ref, model = small_model(), small_model()
    optimizer = torch.optim.Adam(model.net.parameters(), lr=3e-3)
    features, *_ = synthetic_batch(model, 4000, 0)
    with torch.no_grad():
        ref_intent, ref_steer, _ = ref.net(features)
    before = brake_probability(model, features, features[:, 0].numpy() > 0.5)
    for k in range(30):                                   # 30 iterations of fresh on-policy batches
        f, levels, old_logp, adv, _ = synthetic_batch(model, 4000, k + 1)
        stats = ppo_update(model, ref, f, levels, old_logp, adv, optimizer, epochs=4, batch=500, kl_coef=0.01,
                           target_kl=1.0, seed=k)
    after = brake_probability(model, features, features[:, 0].numpy() > 0.5)
    assert after[0] > before[0] + 0.4 and after[1] < before[1]       # braking moved to the states that reward it
    with torch.no_grad():
        intent, steer, _ = model.net(features)
    assert torch.equal(steer, ref_steer)                               # its own network: no gradient reaches it
    # intent shares the trunk with the throttle; distillation holds it at the imitation network's
    assert (torch.softmax(intent, -1) - torch.softmax(ref_intent, -1)).abs().max() < 0.05
    assert stats["epochs_run"] == 4 and 0 <= stats["clip_fraction"] <= 1


def test_ppo_without_advantage_stays_at_the_imitation_policy_and_stops_on_kl():
    ref, model = small_model(), small_model()
    optimizer = torch.optim.Adam(model.net.parameters(), lr=3e-3)
    f, levels, old_logp, adv, _ = synthetic_batch(model, 2000, 0)
    stats = ppo_update(model, ref, f, levels, old_logp, torch.zeros_like(adv), optimizer, epochs=4, batch=500)
    assert stats["kl_ref"] < 1e-3
    pushed = small_model()
    stats = ppo_update(pushed, ref, f, levels, old_logp, adv, torch.optim.Adam(pushed.net.parameters(), lr=0.1),
                       epochs=20, batch=500, kl_coef=0.0, target_kl=0.001)
    assert stats["stopped_early"] and stats["epochs_run"] < 20


def test_critic_learns_episode_returns():
    model = small_model()
    critic = ValueNet.for_model(model, hidden=(64, 64))
    rng = np.random.default_rng(0)
    features = torch.as_tensor(rng.random((3000, 16)).astype(np.float32))
    returns = 2.0 * features[:, 1] - features[:, 2]
    before = explained_variance(batched(critic, features).numpy(), returns.numpy())
    critic_update(critic, features, returns, torch.optim.Adam(critic.parameters(), lr=1e-3), epochs=40, batch=256)
    after = explained_variance(batched(critic, features).numpy(), returns.numpy())
    assert before < 0.5 < 0.9 < after
