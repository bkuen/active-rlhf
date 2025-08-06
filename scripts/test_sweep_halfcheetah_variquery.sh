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
  --exp_name prefppo_variquery \
  --seed $SEED \
  --torch_deterministic True \
  --cuda True \
  --track True \
  --wandb_project_name thesis-benchmark-halfcheetah-tessstttt \
  --wandb_entity bkuen-ludwig-maximilianuniversity-of-munich \
  --wandb_tags prefppo_variquery \
  --env_id HalfCheetah-v5 \
  --total_timesteps 1000000 \
  --num_envs 1 \
  --num_steps 2048 \
  --num_minibatches=32 \
  --update_epochs=10 \
  --replay_buffer_capacity 1000000 \
  --anneal_lr True \
  --norm_adv True \
  --max_grad_norm 0.5 \
  --clip_vloss True \
  --target_kl None \
  --learning_rate 0.0005 \
  --gamma 0.98 \
  --gae_lambda 0.9 \
  --clip_coef 0.3 \
  --ent_coef 0.0006 \
  --vf_coef 0.35 \
  --fragment_length 50 \
  --selector_type variquery \
  --query_schedule linear \
  --total_queries 500 \
  --queries_per_session 10 \
  --sampling_strategy uniform \
  --oversampling_factor 10.0 \
  --reward_net_epochs 5 \
  --reward_net_max_grad_norm 1.0 \
  --reward_net_minibatch_size 32 \
  --reward_net_ensemble_size 3 \
  --reward_net_val_split 0.2 \
  --reward_net_lr 0.00008 \
  --reward_net_weight_decay 0.00001 \
  --reward_net_dropout 0.1 \
  --reward_net_batch_size 32 \
  --reward_net_hidden_dims "[256, 256, 256]" \
  --variquery-vae-lr=7e-4 \
  --variquery-vae-weight-decay=1e-4 \
  --variquery-vae-latent-dim=16 \
  --variquery-vae-hidden-dims "[64, 32]" \
  --variquery-vae-batch-size=4096 \
  --variquery-vae-minibatch-size=256 \
  --variquery-vae-val-batch-size=256 \
  --variquery-vae-num-epochs=1 \
  --variquery-vae-kl-warmup-epochs=2 \
  --variquery_vae_kl_warmup_steps=2 \
  --variquery_vae_early_stopping_patience=20 \
  --variquery-vae-dropout=0.0 \
  --variquery-vae-kl-weight=0.0002 \
  --variquery_vae_noise_sigma=0.1 \
  --variquery-vae-conv-kernel-size 5 \
  --variquery_cluster_size=12 \
  --capture_video False \
  --save_model True \
  --upload_model False \
  --hf_entity ""
