#!/bin/bash
set -x

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
fi

if [ -z "$ALGO" ]; then
    ALGO=PPO-Token-TIS
fi

if [ -z "$DTYPE" ]; then
    DTYPE=float16
fi

if [ -z "$LOSS_AGG_MODE" ]; then
    LOSS_AGG_MODE=seq-mean-token-sum-norm
fi

# Use environment variables for WandB if set, otherwise use defaults
if [ -z "$WANDB_PROJECT" ]; then
    WANDB_PROJECT=precision-rl
fi

if [ -z "$EXP_NAME" ]; then
    EXP_NAME=sanity_test-$DTYPE-$ALGO
fi

if [ -z "$NODES_REQUIRED" ]; then
    NODES_REQUIRED=1
fi

if [ -z "$BATCH_SIZE_MULTIPLIER" ]; then
    BATCH_SIZE_MULTIPLIER=1
fi

# Calculate batch sizes based on multiplier
BASE_TRAIN_BATCH_SIZE=64
BASE_MINI_BATCH_SIZE=16
TRAIN_BATCH_SIZE=$((BASE_TRAIN_BATCH_SIZE * BATCH_SIZE_MULTIPLIER))
MINI_BATCH_SIZE=$((BASE_MINI_BATCH_SIZE * BATCH_SIZE_MULTIPLIER))

echo $MODEL_PATH
echo $ALGO
echo $DTYPE
echo "Nodes: $NODES_REQUIRED"
echo "Train batch size: $TRAIN_BATCH_SIZE, Mini batch size: $MINI_BATCH_SIZE"
echo "${@:1}"

# Train over single or multiple nodes, 8 GPUs per node.
RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-name=fp16_trainer \
    data.train_files=./sanity_test/math_1460.parquet \
    data.val_files=[./sanity_test/aime_2024.parquet,./sanity_test/aime_2025.parquet] \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.policy_loss.loss_mode=$ALGO \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.ref.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.rollout.dtype=$DTYPE \
    critic.model.fsdp_config.dtype=$DTYPE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXP_NAME \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.log_val_generations=20 \
    trainer.val_before_train=True \
    trainer.total_epochs=40 \
    trainer.nnodes=$NODES_REQUIRED \
    trainer.n_gpus_per_node=8 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    algorithm.norm_adv_by_std_in_grpo=False \
    "${@:1}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.use_inductor=False
