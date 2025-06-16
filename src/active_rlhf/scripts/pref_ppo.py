# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Literal

from active_rlhf.algorithms.pref_ppo import Agent, AgentTrainer
from active_rlhf.algorithms.variquery.selector import VARIQuerySelector
import gymnasium as gym
import numpy as np
import torch as th
import tyro
from torch.utils.tensorboard import SummaryWriter

from active_rlhf.data.buffers import ReplayBuffer, PreferenceBuffer, PreferenceBufferBatch
from active_rlhf.data.dataset import PreferenceDataset
from active_rlhf.rewards.reward_nets import PreferenceModel, RewardEnsemble, RewardTrainer
from active_rlhf.queries.selector import RandomSelector, RandomSelectorSimple
from active_rlhf.algorithms.variquery.vae import StateVAE, VAETrainer
from active_rlhf.algorithms.variquery.visualizer import VAEVisualizer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v5"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Reward training
    replay_buffer_capacity: int = 1000000
    """the capacity of the replay buffer"""
    reward_net_epochs: int = 3
    """the number of epochs to train the reward network"""
    reward_net_lr: float = 1e-3
    """the learning rate of the reward network"""
    reward_net_weight_decay: float = 0.0
    """the weight decay of the reward network"""
    reward_net_batch_size: int = 32
    """the batch size of the reward network"""
    reward_net_minibatch_size: int = 32
    """the mini-batch size of the reward network"""
    reward_net_ensemble_size: int = 3
    """the number of ensemble members in the reward network"""
    reward_net_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    """the hidden dimensions of the reward network"""
    reward_net_dropout: float = 0.0
    """the dropout rate of the reward network"""
    reward_net_val_split: float = 0.2
    """the validation split ratio for the reward network training data"""
    query_schedule: str = "linear"
    """the schedule for querying the reward network"""
    total_queries: int = 400
    """the total number of queries to send to the teacher"""
    queries_per_session: int = 10
    """the number of queries to send to the teacher per session"""
    selector_type: Literal["random", "variquery"] = "random"
    """type of selector to use for query selection"""
    oversampling_factor: float = 2.0
    """the oversampling factor for the selector"""
    fragment_length: int = 50
    """length of the fragments"""

    # VARIQuery specific arguments
    variquery_vae_latent_dim: int = 32
    """dimension of the VAE latent space"""
    variquery_vae_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    """hidden dimensions of the VAE encoder/decoder"""
    variquery_vae_lr: float = 1e-3
    """learning rate for the VAE"""
    variquery_vae_weight_decay: float = 1e-4
    """weight decay for the VAE"""
    variquery_vae_batch_size: int = 32
    """batch size for VAE training"""
    variquery_vae_num_epochs: int = 25
    """number of epochs to train the VAE"""
    variquery_vae_dropout: float = 0.1
    """dropout rate for the VAE"""
    variquery_vae_kl_weight: float = 1.0
    """weight of the KL loss term in VAE training"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space=None)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

if __name__ == "__main__":
    args = tyro.cli(Args, config=(tyro.conf.FlagConversionOff,))
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.backends.cudnn.deterministic = args.torch_deterministic

    device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Reward training setup
    replay_buffer = ReplayBuffer(
        capacity=args.replay_buffer_capacity,
        envs=envs,
    )
    
    # Create separate buffers for training and validation
    train_preference_buffer = PreferenceBuffer(
        capacity=int(args.total_queries * (1 - args.reward_net_val_split)),
    )
    val_preference_buffer = PreferenceBuffer(
        capacity=int(args.total_queries * args.reward_net_val_split),
    )


    with th.random.fork_rng(devices=[]):  # empty list=CPU only; pass your GPU devices if needed
        th.manual_seed(args.seed)
        th.cuda.manual_seed_all(args.seed)  # if you're on CUDA

        reward_ensemble = RewardEnsemble(
            obs_dim=envs.single_observation_space.shape[0],
            act_dim=envs.single_action_space.shape[0],
            hidden_dims=args.reward_net_hidden_dims,
            dropout=args.reward_net_dropout,
            ensemble_size=args.reward_net_ensemble_size,
        )

    preference_model = PreferenceModel(reward_ensemble)

    reward_trainer = RewardTrainer(
        preference_model=preference_model,
        writer=writer,
        epochs=args.reward_net_epochs,
        lr=args.reward_net_lr,
        weight_decay=args.reward_net_weight_decay,
        batch_size=args.reward_net_batch_size,
        minibatch_size=args.reward_net_minibatch_size,
    )

    if args.query_schedule == "linear":
        query_schedule = np.linspace(0, 1, (args.total_queries // args.queries_per_session) + 1, endpoint=False)[1:]
        query_schedule = query_schedule * args.total_timesteps
    else:
        raise NotImplementedError(f"Query schedule {args.query_schedule} is not implemented.")

    print(f"Query schedule: {query_schedule}")

    # Initialize selector
    if args.selector_type == "random":
        selector = RandomSelectorSimple()
    elif args.selector_type == "variquery":
        # Initialize VARIQuery selector
        selector = VARIQuerySelector(
            writer=writer,
            reward_ensemble=reward_ensemble,
            fragment_length=args.fragment_length,
            vae_latent_dim=args.variquery_vae_latent_dim,
            vae_hidden_dims=args.variquery_vae_hidden_dims,
            vae_lr=args.variquery_vae_lr,
            vae_weight_decay=args.variquery_vae_weight_decay,
            vae_dropout=args.variquery_vae_dropout,
            vae_batch_size=args.variquery_vae_batch_size,
            vae_num_epochs=args.variquery_vae_num_epochs,
            device=device,
        )
    else:
        raise ValueError(f"Unknown selector type: {args.selector_type}")

    # Agent setup
    agent = Agent(envs).to(device)
    agent_trainer = AgentTrainer(
        agent=agent,
        reward_ensemble=reward_ensemble,
        envs=envs,
        device=device,
        lr=args.learning_rate,
        anneal_lr=args.anneal_lr,
        num_iterations=args.num_iterations,
        num_envs=args.num_envs,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        norm_adv=args.norm_adv,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        seed=args.seed,
    )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    next_query_step = 0

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        agent_trainer.update_learning_rate(iteration)

        # Collect rollout
        rollout_sample, episode_infos = agent_trainer.collect_rollout(global_step, args.num_steps)
        replay_buffer.add_rollout(rollout_sample)

        global_step += args.num_envs * args.num_steps

        # Update policy
        metrics = agent_trainer.update_policy(rollout_sample=rollout_sample, num_steps=args.num_steps)

        # Train reward network if next step is in query schedule. Be careful as we might overstep the query schedule.
        if next_query_step < len(query_schedule) and global_step >= query_schedule[next_query_step]:
            # Sample from replay buffer to get trajectory pairs
            num_pairs = args.queries_per_session if next_query_step != 0 else 32
            reward_samples = replay_buffer.sample(int(num_pairs*args.oversampling_factor))

            # Use selector to get trajectory pairs
            trajectory_pairs = selector.select_pairs(reward_samples, num_pairs=num_pairs, global_step=global_step)
            assert len(trajectory_pairs) == num_pairs

            # Calculate preferences based on returns
            first_returns = trajectory_pairs.first_rews.squeeze().sum(dim=-1)
            second_returns = trajectory_pairs.second_rews.squeeze().sum(dim=-1)
            
            # Create preference tensor where [1,0] means first is preferred
            prefs = th.stack(
                [(first_returns > second_returns).float(),
                 (second_returns >= first_returns).float()],
                dim=1,
            )
            
            # Create preference batch
            preference_batch = PreferenceBufferBatch(
                first_obs=trajectory_pairs.first_obs,
                first_acts=trajectory_pairs.first_acts,
                first_rews=trajectory_pairs.first_rews,
                first_dones=trajectory_pairs.first_dones,
                second_obs=trajectory_pairs.second_obs,
                second_acts=trajectory_pairs.second_acts,
                second_rews=trajectory_pairs.second_rews,
                second_dones=trajectory_pairs.second_dones,
                prefs=prefs
            )
            
            # Split into train and validation based on val_split parameter
            train_size = int(args.reward_net_batch_size * (1 - args.reward_net_val_split))
            train_batch = PreferenceBufferBatch(
                first_obs=preference_batch.first_obs[:train_size],
                first_acts=preference_batch.first_acts[:train_size],
                first_rews=preference_batch.first_rews[:train_size],
                first_dones=preference_batch.first_dones[:train_size],
                second_obs=preference_batch.second_obs[:train_size],
                second_acts=preference_batch.second_acts[:train_size],
                second_rews=preference_batch.second_rews[:train_size],
                second_dones=preference_batch.second_dones[:train_size],
                prefs=preference_batch.prefs[:train_size]
            )
            val_batch = PreferenceBufferBatch(
                first_obs=preference_batch.first_obs[train_size:],
                first_acts=preference_batch.first_acts[train_size:],
                first_rews=preference_batch.first_rews[train_size:],
                first_dones=preference_batch.first_dones[train_size:],
                second_obs=preference_batch.second_obs[train_size:],
                second_acts=preference_batch.second_acts[train_size:],
                second_rews=preference_batch.second_rews[train_size:],
                second_dones=preference_batch.second_dones[train_size:],
                prefs=preference_batch.prefs[train_size:]
            )
            
            # Add to respective buffers
            train_preference_buffer.add(train_batch)
            val_preference_buffer.add(val_batch)

            assert len(train_preference_buffer) <= args.total_queries
            assert len(val_preference_buffer) <= args.total_queries

            # Train reward network if we have enough samples
            if len(train_preference_buffer) > 0 and len(val_preference_buffer) > 0:
                try:
                    reward_trainer.train(train_preference_buffer, val_preference_buffer, global_step)
                except ValueError as e:
                    print(f"Warning: {e}. Skipping reward network training for this iteration.")
                    os.exit(1)

            next_query_step += 1

        # Log metrics
        writer.add_scalar("charts/learning_rate", agent_trainer.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", metrics.v_loss, global_step)
        writer.add_scalar("losses/policy_loss", metrics.pg_loss, global_step)
        writer.add_scalar("losses/entropy", metrics.entropy_loss, global_step)
        writer.add_scalar("losses/old_approx_kl", metrics.old_approx_kl, global_step)
        writer.add_scalar("losses/approx_kl", metrics.approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", metrics.clipfrac, global_step)
        writer.add_scalar("losses/explained_variance", metrics.explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Log episode info
        for info in episode_infos:
            writer.add_scalar("charts/episodic_return", info.episode_reward, info.relative_step)
            writer.add_scalar("charts/episodic_length", info.episode_length, info.relative_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        th.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        # from cleanrl_utils.evals.ppo_eval import evaluate
        #
        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     args.env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=Agent,
        #     device=device,
        #     gamma=args.gamma,
        # )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub
        #
        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()