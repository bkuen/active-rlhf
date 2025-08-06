SEEDS=(
  403021432
  2001339610
  109214983
  1441960350
  57898117
  1949773967
  677630646
  422600776
  406989952
  27474278
)

# ensure SLURM_ARRAY_TASK_ID is set and in [1..10]
: "${SLURM_ARRAY_TASK_ID:?Need SLURM_ARRAY_TASK_ID (1–10)}"
if [ "$SLURM_ARRAY_TASK_ID" -lt 1 ] || [ "$SLURM_ARRAY_TASK_ID" -gt "${#SEEDS[@]}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID must be between 1 and ${#SEEDS[@]}" >&2
  exit 1
fi

# pick the corresponding seed
IDX=$((SLURM_ARRAY_TASK_ID - 1))
SEED="${SEEDS[$IDX]}"

poetry run python3.10 src/active_rlhf/scripts/pref_ppo.py \
  --exp_name prefppo_hybrid_prio \
  --seed $SEED \
  --torch_deterministic True \
  --cuda True \
  --track True \
  --wandb_project_name thesis-benchmark-hopper \
  --wandb_entity bkuen-ludwig-maximilianuniversity-of-munich \
  --wandb_tags prefppo_hybrid_prio \
  --env_id Hopper-v5 \
  --total_timesteps 1000000 \
  --num_envs 1 \
  --num_steps 2048 \
  --replay_buffer_capacity 1000000 \
  --anneal_lr True \
  --norm_adv True \
  --max_grad_norm 0.5 \
  --clip_vloss True \
  --target_kl None \
  --learning_rate 3e-5 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_coef 0.3 \
  --ent_coef 3e-4 \
  --vf_coef 0.55 \
  --fragment_length 75 \
  --selector_type hybrid \
  --query_schedule linear \
  --total_queries 500 \
  --queries_per_session 10 \
  --sampling_strategy priority \
  --oversampling_factor 10.0 \
  --reward_net_epochs 5 \
  --reward_net_max_grad_norm 1.0 \
  --reward_net_minibatch_size 32 \
  --reward_net_ensemble_size 3 \
  --reward_net_val_split 0.2 \
  --reward_net_lr 8e-5 \
  --reward_net_weight_decay 5e-6 \
  --reward_net_dropout 0.2 \
  --reward_net_batch_size 32 \
  --reward_net_hidden_dims "[256, 256, 256]" \
  --variquery-vae-lr=0.002 \
  --variquery-vae-weight-decay=2e-05 \
  --variquery-vae-latent-dim=64 \
  --variquery-vae-hidden-dims="[64, 32]" \
  --variquery-vae-batch-size=2048 \
  --variquery-vae-minibatch-size=512 \
  --variquery-vae-num-epochs=5 \
  --variquery-vae-dropout=0.0 \
  --variquery-vae-kl-weight=0.0 \
  --variquery-vae-noise-sigma=0.0 \
  --variquery_cluster_size=10 \
  --hybrid_dpp_beta 0.5 \
  --hybrid_dpp_min_q 0.6 \
  --hybrid_dpp_temp_q 0.7 \
  --capture_video False \
  --save_model True \
  --upload_model False \
  --hf_entity ""