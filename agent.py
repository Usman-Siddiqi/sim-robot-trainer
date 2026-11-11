"""
agent.py

Author: Usman Siddiqi

The reinforcement learning agent: a PPO (Proximal Policy Optimization)
implementation written for massively parallel simulation.

  * ActorCritic  - the policy (actor) and value (critic) MLPs plus a running
                   observation normaliser
  * PPO          - rollout storage, GAE advantages and the clipped PPO update
                   with a KL-adaptive learning rate

Everything is kept on one torch device. On a CUDA machine the whole batch of
robots is evaluated in a single GPU forward pass per control step and the PPO
update runs on large GPU minibatches, which is where the RTX 4050 earns its keep.
"""

import torch
import torch.nn as nn

from config import PPOCfg


def get_device(name="auto"):
    """'auto' picks the GPU when PyTorch can see one."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def print_device_info(device):
    print(f"PyTorch {torch.__version__} | device: {device}", end="")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f" ({props.name}, {props.total_memory / 2**30:.1f} GB)")
    else:
        print(f" ({torch.get_num_threads()} threads)")
        if not torch.cuda.is_available():
            print("  note: no CUDA GPU visible to PyTorch - see the README to install the CUDA build of torch")


class RunningNorm(nn.Module):
    """Normalises observations with a running mean / variance (frozen at inference)."""

    def __init__(self, size, eps=1e-5, clip=10.0):
        super().__init__()
        self.eps, self.clip = eps, clip
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))
        # entries taken over from another policy keep the statistics that policy was trained with
        self.register_buffer("frozen", torch.zeros(size, dtype=torch.bool))

    @torch.no_grad()
    def update(self, x):
        batch_mean, batch_var, n = x.mean(dim=0), x.var(dim=0, unbiased=False), x.shape[0]
        total = self.count + n
        delta = batch_mean - self.mean
        mean = self.mean + delta * n / total
        var = (self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total) / total
        self.mean.copy_(torch.where(self.frozen, self.mean, mean))
        self.var.copy_(torch.where(self.frozen, self.var, var))
        self.count.copy_(total)

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.eps), -self.clip, self.clip)


def mlp(sizes, activation=nn.ELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions, actor_hidden=None, critic_hidden=None, init_noise_std=None):
        super().__init__()
        actor_hidden = list(actor_hidden or PPOCfg.actor_hidden)
        critic_hidden = list(critic_hidden or PPOCfg.critic_hidden)
        init_noise_std = init_noise_std or PPOCfg.init_noise_std
        self.num_obs, self.num_actions = num_obs, num_actions
        self.actor_hidden, self.critic_hidden = actor_hidden, critic_hidden
        self.normalizer = RunningNorm(num_obs)
        self.actor = mlp([num_obs] + actor_hidden + [num_actions])
        self.critic = mlp([num_obs] + critic_hidden + [1])
        self.log_std = nn.Parameter(torch.full((num_actions,), float(init_noise_std)).log())

    def distribution(self, norm_obs):
        mean = self.actor(norm_obs)
        return mean, self.log_std.exp().expand_as(mean)

    def value(self, norm_obs):
        return self.critic(norm_obs).squeeze(-1)

    @torch.no_grad()
    def act_inference(self, obs):
        """Deterministic action (the mean of the policy) from raw observations."""
        return self.actor(self.normalizer(obs))


def gaussian_log_prob(actions, mean, std):
    var = std ** 2
    return (-((actions - mean) ** 2) / (2 * var) - std.log() - 0.9189385332046727).sum(dim=-1)


class PPO:
    def __init__(self, policy, num_envs, device, cfg=PPOCfg):
        self.policy = policy.to(device)
        self.device = device
        self.cfg = cfg
        self.lr = cfg.learning_rate
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.train_actor = True             # False: only the value function learns (warm-up after a transfer)

        T, N, A, O = cfg.num_steps, num_envs, policy.num_actions, policy.num_obs
        self.obs = torch.zeros(T, N, O, device=device)
        self.actions = torch.zeros(T, N, A, device=device)
        self.rewards = torch.zeros(T, N, device=device)
        self.dones = torch.zeros(T, N, device=device)
        self.values = torch.zeros(T, N, device=device)
        self.log_probs = torch.zeros(T, N, device=device)
        self.means = torch.zeros(T, N, A, device=device)
        self.stds = torch.zeros(T, N, A, device=device)
        self.step = 0

    # ------------------------------------------------------------- rollouts
    @torch.no_grad()
    def act(self, obs):
        """Sample actions for a batch of raw observations and remember everything PPO needs."""
        self.policy.normalizer.update(obs)
        norm_obs = self.policy.normalizer(obs)
        mean, std = self.policy.distribution(norm_obs)
        actions = mean + std * torch.randn_like(mean)
        t = self.step
        self.obs[t] = norm_obs
        self.actions[t] = actions
        self.values[t] = self.policy.value(norm_obs)
        self.log_probs[t] = gaussian_log_prob(actions, mean, std)
        self.means[t] = mean
        self.stds[t] = std
        return actions

    @torch.no_grad()
    def record(self, rewards, dones, time_outs):
        """Store the result of the step taken with the last act() call."""
        t = self.step
        # Episodes cut off by a timeout did not really end: add the value of the state we stopped in
        self.rewards[t] = rewards + self.cfg.gamma * self.values[t] * time_outs
        self.dones[t] = dones
        self.step += 1

    @torch.no_grad()
    def _compute_returns(self, last_obs):
        cfg = self.cfg
        last_value = self.policy.value(self.policy.normalizer(last_obs))
        advantage = torch.zeros_like(last_value)
        returns = torch.zeros_like(self.rewards)
        for t in reversed(range(cfg.num_steps)):
            next_value = last_value if t == cfg.num_steps - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + cfg.gamma * next_value * not_done - self.values[t]
            advantage = delta + cfg.gamma * cfg.lam * not_done * advantage
            returns[t] = advantage + self.values[t]
        advantages = returns - self.values
        return returns, (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # --------------------------------------------------------------- update
    def update(self, last_obs):
        """One PPO update from the stored rollout. Returns a dict of losses for logging."""
        cfg = self.cfg
        returns, advantages = self._compute_returns(last_obs)
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        obs, actions, old_values = flat(self.obs), flat(self.actions), flat(self.values)
        old_log_probs, old_means, old_stds = flat(self.log_probs), flat(self.means), flat(self.stds)
        returns, advantages = flat(returns), flat(advantages)

        batch = obs.shape[0]
        mb_size = batch // cfg.num_minibatches
        stats = {"value_loss": 0.0, "surrogate_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        updates = 0
        for _ in range(cfg.num_epochs):
            perm = torch.randperm(batch, device=self.device)
            for i in range(cfg.num_minibatches):
                idx = perm[i * mb_size:(i + 1) * mb_size]
                mean, std = self.policy.distribution(obs[idx])
                value = self.policy.value(obs[idx])
                log_prob = gaussian_log_prob(actions[idx], mean, std)
                entropy = (0.5 + 0.9189385332046727 + std.log()).sum(dim=-1).mean()

                # adapt the learning rate so each update changes the policy by about desired_kl
                with torch.no_grad():
                    kl = (torch.log(std / old_stds[idx])
                          + (old_stds[idx] ** 2 + (old_means[idx] - mean) ** 2) / (2.0 * std ** 2) - 0.5).sum(dim=-1).mean()
                    if cfg.desired_kl is not None and self.train_actor:
                        if kl > cfg.desired_kl * 2.0:
                            self.lr = max(1e-5, self.lr / 1.5)
                        elif 0.0 < kl < cfg.desired_kl / 2.0:
                            self.lr = min(1e-2, self.lr * 1.5)
                        for group in self.optimizer.param_groups:
                            group["lr"] = self.lr

                ratio = torch.exp(log_prob - old_log_probs[idx])
                adv = advantages[idx]
                surrogate = -torch.min(ratio * adv,
                                       torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * adv).mean()
                value_clipped = old_values[idx] + (value - old_values[idx]).clamp(-cfg.clip_param, cfg.clip_param)
                value_loss = torch.max((value - returns[idx]) ** 2, (value_clipped - returns[idx]) ** 2).mean()
                loss = cfg.value_loss_coef * value_loss
                if self.train_actor:
                    loss = loss + surrogate - cfg.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                stats["value_loss"] += value_loss.item()
                stats["surrogate_loss"] += surrogate.item()
                stats["entropy"] += entropy.item()
                stats["kl"] += kl.item()
                updates += 1

        self.step = 0
        stats = {k: v / updates for k, v in stats.items()}
        stats["learning_rate"] = self.lr
        stats["noise_std"] = self.policy.log_std.exp().mean().item()
        return stats


# ------------------------------------------------------------- checkpoints
def save_checkpoint(path, policy, ppo=None, iteration=0, extra=None):
    data = {
        "policy": policy.state_dict(),
        "num_obs": policy.num_obs,
        "num_actions": policy.num_actions,
        "actor_hidden": policy.actor_hidden,
        "critic_hidden": policy.critic_hidden,
        "iteration": iteration,
    }
    if ppo is not None:
        data["optimizer"] = ppo.optimizer.state_dict()
        data["learning_rate"] = ppo.lr
    data.update(extra or {})
    torch.save(data, path)


def load_policy(path, device):
    """Load a checkpoint. Returns (policy, checkpoint dict)."""
    data = torch.load(path, map_location=device, weights_only=True)
    policy = ActorCritic(data["num_obs"], data["num_actions"], data["actor_hidden"], data["critic_hidden"])
    state = dict(data["policy"])
    state.setdefault("normalizer.frozen", torch.zeros(data["num_obs"], dtype=torch.bool))
    policy.load_state_dict(state)
    return policy.to(device), data


def transfer_policy(path, num_obs, num_actions, device, arm_slice=None, arm_std=None):
    """
    Start a new policy from one trained on another task whose observation is the first part of
    the new one (the sorting observation starts with the hiking observation).

    The actor is copied; the weights of the extra inputs start at zero, so at first the new policy
    behaves exactly like the old one. The value function starts from scratch (the rewards differ).
    """
    old, data = load_policy(path, device)
    if old.num_actions != num_actions or old.num_obs > num_obs:
        raise ValueError(f"{path} cannot be used as a starting point for this task")
    policy = ActorCritic(num_obs, num_actions, old.actor_hidden, data["critic_hidden"]).to(device)
    k = old.num_obs
    with torch.no_grad():
        state = old.actor.state_dict()
        first = policy.actor[0].weight
        first.zero_()
        first[:, :k] = state.pop("0.weight")
        policy.actor[0].bias.copy_(state.pop("0.bias"))
        policy.actor.load_state_dict(state, strict=False)
        policy.log_std.copy_(old.log_std)
        if arm_slice is not None and arm_std is not None:
            policy.log_std[arm_slice] = torch.clamp(policy.log_std[arm_slice], min=float(torch.tensor(arm_std).log()))
        policy.normalizer.mean[:k] = old.normalizer.mean
        policy.normalizer.var[:k] = old.normalizer.var
        policy.normalizer.frozen[:k] = True
    return policy, data
