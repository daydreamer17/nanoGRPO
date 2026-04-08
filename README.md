# nanoGRPO

最小但可观察趋势的 `TRL GRPO` 示例，目标配置是：

- 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 数据：`openai/gsm8k`
- 训练：`GRPOTrainer + LoRA`
- 生成：双卡 `vLLM server mode`
- 日志：`wandb` 在线同步
- 缓存与输出：统一放到 `/root/autodl-tmp`

## 文件

- `setup_env.sh`: 创建独立 conda 环境并安装固定版本依赖
- `train_grpo_gsm8k.py`: 训练入口
- `grpo_gsm8k_utils.py`: 数据预处理、答案解析、缓存目录与 W&B 校验辅助函数
- `accelerate_config.yaml`: 单卡 trainer 配置

## 1. 创建环境

当前机器上的 `torch 2.5.1+cu124` 会对 `RTX 5090 (sm_120)` 给出不兼容警告，所以请先创建独立环境：

```bash
cd /root/nanoGRPO
bash setup_env.sh
conda activate grpo-qwen05b
```

如果你想换环境名：

```bash
bash setup_env.sh my-grpo-env
conda activate my-grpo-env
```

## 2. 配置 W&B

这个示例默认强制在线同步，不会自动退回本地日志。训练前先登录：

```bash
wandb login
```

或者提前设置：

```bash
export WANDB_API_KEY=...
```

默认项目名是 `nano-grpo-qwen05b`。

## 3. 缓存与输出目录

脚本会自动创建并使用这些目录：

- `HF_HOME=/root/autodl-tmp/hf`
- `HF_DATASETS_CACHE=/root/autodl-tmp/hf/datasets`
- `WANDB_DIR=/root/autodl-tmp/wandb`
- `output_dir=/root/autodl-tmp/outputs/nano-grpo-qwen05b`

不需要你手动 export，脚本会自动设置。

## 4. 预检训练

这一步只验证：

- 数据下载和预处理
- reward 函数
- LoRA 包装
- checkpoint 保存
- W&B run 创建

命令如下：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file accelerate_config.yaml \
  train_grpo_gsm8k.py \
  --train_slice 'train[:32]' \
  --eval_size 32 \
  --max_steps 1 \
  --use_vllm false \
  --wandb_run_name preflight-grpo-gsm8k
```

## 5. 启动 vLLM 服务

正式训练时，把 GPU1 专门留给 vLLM：

```bash
CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1
```

## 6. 正式训练

GPU0 跑 trainer，GPU1 跑 vLLM：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file accelerate_config.yaml \
  train_grpo_gsm8k.py \
  --dataset_root /root/autodl-tmp \
  --train_slice 'train[:512]' \
  --eval_size 32 \
  --max_steps 100 \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port 8000 \
  --wandb_project nano-grpo-qwen05b
```

训练结束后会在下面目录保存最终 adapter：

```bash
/root/autodl-tmp/outputs/nano-grpo-qwen05b/final_adapter
```

## 训练配置摘要

- `train[:512]`
- `max_steps=100`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=4`
- `num_generations=4`
- `learning_rate=1e-5`
- `max_prompt_length=256`
- `max_completion_length=160`
- `beta=0.0`
- `temperature=0.7`
- `LoRA r=16 alpha=32 dropout=0.05`

## 实现说明

- `format_reward`: 只要最后能解析出 `Final answer: <number>`，奖励 `0.2`
- `answer_reward`: 最终数字与 gold 完全一致，奖励 `1.0`
- `max_prompt_length` 在 `TRL 1.0.0` 里没有单独的 `GRPOConfig` 字段，所以这个仓库里把它实现成：
  - 对数据集做 prompt token 长度检查
  - 同时把 `vllm_max_model_length` 设为 `max_prompt_length + max_completion_length`

## 常见问题

### 1. 为什么不直接在 base 环境安装？

因为当前 base 环境已经有不兼容 5090 的 `torch 2.5.1+cu124`，直接覆盖更容易把别的东西带坏。

### 2. 为什么锁到 `vllm==0.17.1`？

因为 `TRL` 官方 vLLM 集成文档当前写的是支持 `0.10.2` 到 `0.17.1`。

### 3. 为什么 reward 不直接用内建 `accuracy_reward`？

这个示例需要固定输出 `Final answer: <number>` 的格式约束，而且 GSM8K 的答案抽取也想自己控制，所以这里用了自定义 reward。
