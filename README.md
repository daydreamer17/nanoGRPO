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
- `train_sft_gsm8k.py`: `SFT + LoRA` 训练入口
- `train_grpo_gsm8k.py`: 训练入口
- `merge_lora_adapter.py`: 把 LoRA adapter 合并为独立模型目录
- `grpo_gsm8k_utils.py`: 数据预处理、答案解析、缓存目录与 W&B 校验辅助函数
- `dense_reward_v2.py`: `Dense Reward V2` 的独立实现
- `prepare_sft_to_grpo.sh`: 一键跑完 `SFT -> merge` 的准备脚本
- `accelerate_config.yaml`: 单卡 trainer 配置
- `run_second_round_eval.sh`: 一次性跑完第二轮 checkpoint sweep 的脚本
- `DENSE_REWARD_V2.md`: 下一版更 dense reward 设计文档
- `SFT_TO_GRPO_CHAIN.md`: `SFT + LoRA -> GRPO` 链路说明
- `NEXT_ITERATION_PLAN_AFTER_SFT_TO_GRPO_20260415.md`: 当前最新一轮 `SFT -> GRPO` 复盘与下一轮建议

## SFT + LoRA -> GRPO

现在仓库里已经支持一条更稳的链路：

1. `train_sft_gsm8k.py` 先做 `SFT + LoRA`
2. `merge_lora_adapter.py` 把 `SFT adapter` 合并成新的本地 base model
3. `train_grpo_gsm8k.py --model_name <merged_model_dir>` 再从这个起点继续做 `GRPO`

如果你要直接走这条链路，优先看 [SFT_TO_GRPO_CHAIN.md](SFT_TO_GRPO_CHAIN.md)。  
其中最省事的起手命令是：

```bash
cd /root/nanoGRPO
GPU_ID=0 \
TRAIN_SLICE='train[:1000]' \
MAX_STEPS=200 \
EVAL_SIZE=64 \
OFFLINE=true \
RUN_NAME=sft-gsm8k-qwen25-05b-1000 \
bash prepare_sft_to_grpo.sh
```

## 最新结果

截至目前，这条线里最值得关注的两个结果是：

- 纯 `SFT`:
  - `train[:1000]`
  - `34 / 128`
  - `format_hits = 116 / 128`
- `SFT -> GRPO`:
  - run: `grpo-gsm8k-qwen25-05b-sftinit-1000-lr15e6`
  - best checkpoint: `checkpoint-100`
  - `38 / 128`

这说明：

- `SFT` 已经能把格式和基础答题能力拉到一个很强的起点
- `GRPO` 在这个起点上还能继续带来小幅但真实的提升

当前这条路线下，优先推荐保留：

- `SFT merged_model`: `/root/autodl-tmp/outputs/nano-grpo-qwen05b/sft-gsm8k-qwen25-05b-1000/merged_model`
- best GRPO adapter: `/root/autodl-tmp/outputs/nano-grpo-qwen05b/grpo-gsm8k-qwen25-05b-sftinit-1000-lr15e6/checkpoint-100`

下一轮更具体的建议见 [NEXT_ITERATION_PLAN_AFTER_SFT_TO_GRPO_20260415.md](NEXT_ITERATION_PLAN_AFTER_SFT_TO_GRPO_20260415.md)。

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
- `TRANSFORMERS_CACHE=/root/autodl-tmp/hf/transformers`
- `WANDB_DIR=/root/autodl-tmp/wandb`
- `experiment_dir=/root/autodl-tmp/outputs/nano-grpo-qwen05b`

训练脚本会自动设置这些变量，但 `trl vllm-serve` 不会读取训练脚本里的 Python 逻辑。
如果你的容器访问 Hugging Face 不稳定，建议先把模型缓存预热到本地，再用本地快照路径启动 vLLM。

从现在开始，训练默认会为每个 run 单独创建输出子目录：

- `experiment_dir=/root/autodl-tmp/outputs/nano-grpo-qwen05b`
- `run_dir=/root/autodl-tmp/outputs/nano-grpo-qwen05b/<wandb_run_name>`

这样 checkpoint、summary 和最终 adapter 不会再和别的 run 混在一起。

先拿到本地模型快照路径：

```bash
MODEL_SNAPSHOT=$(cat /root/autodl-tmp/hf/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/refs/main)
echo /root/autodl-tmp/hf/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/$MODEL_SNAPSHOT
```

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
  --offline true \
  --use_vllm false \
  --wandb_run_name preflight-grpo-gsm8k
```

## 5. 启动 vLLM 服务

正式训练时，把 GPU1 专门留给 vLLM：

```bash
MODEL_SNAPSHOT=$(cat /root/autodl-tmp/hf/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/refs/main)

CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_HOME=/root/autodl-tmp/hf \
TRANSFORMERS_CACHE=/root/autodl-tmp/hf/transformers \
trl vllm-serve \
  --model /root/autodl-tmp/hf/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/$MODEL_SNAPSHOT \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1
```

如果你确认联网稳定，也可以把 `--model` 改回 `Qwen/Qwen2.5-0.5B-Instruct`，但在当前环境里优先推荐本地快照方式。

## 6. 正式训练

GPU0 跑 trainer，GPU1 跑 vLLM。下面这条命令已经切到更稳的“第二轮默认配置”：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file accelerate_config.yaml \
  train_grpo_gsm8k.py \
  --dataset_root /root/autodl-tmp \
  --train_slice 'train[:512]' \
  --eval_size 32 \
  --max_steps 200 \
  --max_completion_length 160 \
  --learning_rate 5e-6 \
  --save_steps 10 \
  --save_total_limit 20 \
  --offline true \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port 8000 \
  --wandb_project nano-grpo-qwen05b
```

如果你想继续复现实验历史，这条命令默认仍然使用旧版两段式 reward，也就是：

- `format_reward`
- `answer_reward`

训练结束后会在对应 run 目录里保存最终 adapter：

```bash
/root/autodl-tmp/outputs/nano-grpo-qwen05b/<wandb_run_name>/final_adapter
```

每次训练还会额外写两类结果文件：

- run 内记录：`/root/autodl-tmp/outputs/nano-grpo-qwen05b/<wandb_run_name>/run_results.jsonl`
- run 内摘要：`/root/autodl-tmp/outputs/nano-grpo-qwen05b/<wandb_run_name>/run_summaries/`
- 顶层聚合记录：`/root/autodl-tmp/outputs/nano-grpo-qwen05b/run_results.jsonl`

## 训练配置摘要

- `train[:512]`
- `max_steps=200`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=4`
- `num_generations=4`
- `learning_rate=5e-6`
- `max_prompt_length=256`
- `max_completion_length=160`
- `save_steps=10`
- `save_total_limit=20`
- `beta=0.0`
- `temperature=0.7`
- `LoRA r=16 alpha=32 dropout=0.05`

这组默认值的目的不是追求单次最快，而是让 checkpoint sweep 更稳定：

- 更低学习率，减小后半程抖动
- 更长 completion，上到 `Final answer` 前不容易被截断
- 更频繁保存，方便找中途的甜点 checkpoint
- 更高的 `save_total_limit`，避免做 sweep 时 checkpoint 被自动清掉

## 结果对比文件

- 每次 `train_grpo_gsm8k.py` 跑完会追加一条训练摘要到 `run_results.jsonl`
- 每次 `smoke_eval.py` 跑完也会追加一条评估摘要到 `run_results.jsonl`
- 对应的单次 JSON 会写到 `run_summaries/`，文件名里带时间戳，方便横向比不同 run 和不同 checkpoint

## Checkpoint Sweep

推荐在每轮训练后都跑一次 checkpoint sweep，不要默认 `final_adapter` 或最后一个 checkpoint 最好。

现在两个 sweep 脚本都会自动从当前 run 的 `run_metadata.json` 里读取 `model_name`。
这意味着：

- 普通 `base -> GRPO` run 会自动用原始 base model 做对比
- `SFT -> GRPO` run 会自动用 `SFT merged_model` 做对比

也就是说，当前版本不再需要你手动给 sweep 脚本补 `--model_name`。

单个 checkpoint 评估：

```bash
CUDA_VISIBLE_DEVICES=0 python smoke_eval.py \
  --dataset_root /root/autodl-tmp \
  --output_dir /root/autodl-tmp/outputs/nano-grpo-qwen05b/<wandb_run_name> \
  --adapter_path /root/autodl-tmp/outputs/nano-grpo-qwen05b/<wandb_run_name>/checkpoint-100 \
  --eval_size 32 \
  --offline true \
  --write_jsonl false
```

第二轮推荐直接用一键脚本：

```bash
GPU_ID=0 EVAL_SIZE=64 bash run_second_round_eval.sh
```

如果你想显式指定某个 run 目录，也可以：

```bash
RUN_DIR=/root/autodl-tmp/outputs/nano-grpo-qwen05b/<wandb_run_name> \
GPU_ID=0 EVAL_SIZE=64 \
bash run_second_round_eval.sh
```

默认会评估这些 checkpoint：

- `80`
- `100`
- `110`
- `120`
- `130`
- `140`
- `150`
- `160`
- `200`
- `final_adapter`

所有结果会同时写到：

- 当前 run 的 `run_results.jsonl`
- 顶层聚合文件 `/root/autodl-tmp/outputs/nano-grpo-qwen05b/run_results.jsonl`

## 7. 第三轮推荐训练

第三轮推荐显式切到 `Dense Reward V2`，不要依赖默认值：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file accelerate_config.yaml \
  train_grpo_gsm8k.py \
  --dataset_root /root/autodl-tmp \
  --train_slice 'train[:512]' \
  --eval_size 32 \
  --reward_scheme dense_v2 \
  --max_steps 160 \
  --max_completion_length 160 \
  --learning_rate 5e-6 \
  --save_steps 10 \
  --save_total_limit 20 \
  --offline true \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port 8000 \
  --wandb_project nano-grpo-qwen05b
```

这条第三轮命令的核心变化只有一个：

- `--reward_scheme dense_v2`

其余超参基本沿用第二轮里已经验证过有效的配置，并把训练长度收回到 `160 step`。

## 实现说明

- `format_reward`: 只要最后能解析出 `Final answer: <number>`，奖励 `0.2`
- `answer_reward`: 最终数字与 gold 完全一致，奖励 `1.0`
- `reward_scheme` 现在支持两种模式：
  - `baseline`: 旧版两段式 reward
  - `dense_v2`: 更 dense 的五段式 reward，实现见 `dense_reward_v2.py`
- system prompt 会额外限制“最多 3 行短推理，最后一行之后不能再加内容”，这样更容易把训练信号集中到格式和最终答案上
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

### 4. 为什么 `trl vllm-serve` 会报模型下载失败，但训练脚本有时能跑？

因为训练脚本会在运行时自动把 Hugging Face 缓存切到 `/root/autodl-tmp`，而 `trl vllm-serve` 是独立命令，不会自动继承那段 Python 逻辑。
如果 shell 里没显式设置 `HF_HOME` / `TRANSFORMERS_CACHE`，它会退回默认缓存目录；一旦此时容器又连不上 Hugging Face，就会报 “couldn't find them in the cached files”。

### 5. 为什么训练命令也建议加 `--offline true`？

因为 `datasets` 和 `transformers` 即使已经有本地缓存，默认仍可能先访问 Hub 做一次元数据检查。
在你的容器里，Hugging Face 网络有时不稳定，所以只要数据集和模型已经缓存到 `/root/autodl-tmp`，训练阶段就更适合直接走离线模式。
