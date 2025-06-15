from dataclasses import dataclass
from typing import List, Tuple, TypedDict
import torch as th
import torch.nn as nn
from gymnasium.vector import SyncVectorEnv
from torch.distributions.normal import Normal
import numpy as np

from active_rlhf.data.buffers import RolloutBuffer, RolloutBufferSample
from active_rlhf.rewards.reward_nets import RewardEnsemble


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(th.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = th.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

@dataclass
class AgentMetrics:
    pg_loss: float
    v_loss: float
    entropy_loss: float
    old_approx_kl: float
    approx_kl: float
    clipfrac: float
    explained_var: float

@dataclass
class RolloutEpisodeInfo:
    relative_step: int
    episode_reward: float
    episode_length: int

class AgentTrainer:
    def __init__(self, 
                 agent: Agent,
                 reward_ensemble: RewardEnsemble,
                 envs: SyncVectorEnv, 
                 device: str, 
                 lr: float, 
                 anneal_lr: bool, 
                 num_iterations: int, 
                 num_envs: int, 
                 update_epochs: int, 
                 batch_size: int, 
                 minibatch_size: int, 
                 gamma: float, 
                 gae_lambda: float, 
                 norm_adv: bool,
                 clip_coef: float,
                 clip_vloss: bool,
                 ent_coef: float,
                 vf_coef: float,
                 max_grad_norm: float,
                 target_kl: float,
                 seed: int):
        self.agent = agent
        self.reward_ensemble = reward_ensemble
        self.envs = envs
        self.device = device
        self.lr = lr
        self.anneal_lr = anneal_lr
        self.num_iterations = num_iterations
        self.num_envs = num_envs
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.optimizer = th.optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)
        self.next_obs, _ = envs.reset(seed=seed)
        self.next_obs = th.Tensor(self.next_obs).to(device)
        self.next_done = th.zeros(num_envs).to(device)

    def update_learning_rate(self, iteration: int):
        if self.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.num_iterations
            lrnow = frac * self.lr
            self.optimizer.param_groups[0]["lr"] = lrnow

    def collect_rollout(self, global_step: int, num_steps: int) -> Tuple[RolloutBufferSample, List[RolloutEpisodeInfo]]:
        episode_infos = []

        rollout_buffer = RolloutBuffer(num_steps=num_steps, num_envs=self.num_envs, envs=self.envs, device=self.device)
        for step in range(0, num_steps):
            global_step += self.num_envs

            # ALGO LOGIC: action logic
            with th.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
                value = value.flatten()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            rollout_buffer.store(step=step, 
                                 obs=self.next_obs,
                                 done=self.next_done,
                                 action=action, 
                                 logprob=logprob, 
                                 ground_truth_reward=th.tensor(reward).to(self.device).view(-1),
                                 value=value)

            next_done = np.logical_or(terminations, truncations)

            self.next_obs, self.next_done = th.Tensor(next_obs).to(self.device), th.Tensor(next_done).to(self.device)

            if infos and "episode" in infos:
                episode_infos.append(RolloutEpisodeInfo(
                    relative_step=global_step,
                    episode_reward=infos["episode"]["r"],
                    episode_length=infos["episode"]["l"]
                ))

        return rollout_buffer.get_batch(), episode_infos

    def update_policy(self, rollout_sample: RolloutBufferSample, num_steps: int):
        advantages, returns = self._compute_gaes(rollout_sample, num_steps)

        b_obs = rollout_sample.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = rollout_sample.logprobs.reshape(-1)
        b_actions = rollout_sample.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = rollout_sample.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with th.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + th.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return AgentMetrics(
            pg_loss=pg_loss.item(),
            v_loss=v_loss.item(),
            entropy_loss=entropy_loss.item(),
            old_approx_kl=old_approx_kl.item(),
            approx_kl=approx_kl.item(),
            clipfrac=np.mean(clipfracs),
            explained_var=explained_var
        )

    def _compute_gaes(self, rollout_sample: RolloutBufferSample, num_steps: int):
        with th.no_grad():
            obs = rollout_sample.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            acts = rollout_sample.actions.reshape((-1,) + self.envs.single_action_space.shape)
            reward_preds = self.reward_ensemble.mean_reward(obs, acts).unsqueeze(-1)

            print("Reward predictions shape:", reward_preds.shape)
            print("Ground truth rewards shape:", rollout_sample.ground_truth_rewards.shape)

            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            advantages = th.zeros_like(reward_preds).to(self.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - rollout_sample.dones[t + 1]
                    nextvalues = rollout_sample.values[t + 1]
                delta = reward_preds[t] + self.gamma * nextvalues * nextnonterminal - rollout_sample.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + rollout_sample.values

        return advantages, returns
