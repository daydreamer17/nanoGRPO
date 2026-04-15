# SFT + LoRA -> GRPO

这条链路的目标是先用 `SFT + LoRA` 给 `Qwen/Qwen2.5-0.5B-Instruct` 一个更稳的数学与输出格式起点，再从这个起点继续做 `GRPO`。

## 为什么先做 SFT

- `SFT` 提供更稠密的监督信号，先把题型、短推理风格和 `Final answer: <number>` 格式学稳
- 之后 `GRPO` 不需要把太多训练量浪费在“学会按格式输出”上
- 对 `0.5B` 这种小模型，这通常比“直接从 base 做 RL”更稳

## 产物关系

这一链路会产出三层对象：

- `SFT final_adapter`: 监督微调得到的 LoRA adapter
- `merged_model`: 把 `SFT final_adapter` 合并回 base 之后的本地模型目录
- `GRPO adapter`: 以 `merged_model` 为 base 继续 GRPO 后得到的新 LoRA adapter

最终评估时：

- 看纯 `SFT` 效果：`base model + SFT final_adapter`
- 看 `SFT -> GRPO` 效果：`merged_model + GRPO adapter`

## 第 1 步：训练 SFT + LoRA

最快的方式是直接用准备脚本，它会在训练结束后自动做 merge：

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

如果你想手动分两步跑，也可以先单独训练：

```bash
CUDA_VISIBLE_DEVICES=0 python train_sft_gsm8k.py \
  --dataset_root /root/autodl-tmp \
  --train_slice 'train[:1000]' \
  --eval_size 64 \
  --max_steps 200 \
  --max_seq_length 512 \
  --learning_rate 2e-5 \
  --save_steps 20 \
  --save_total_limit 10 \
  --offline true \
  --wandb_project nano-grpo-qwen05b \
  --wandb_run_name sft-gsm8k-qwen25-05b-1000
```

默认会生成：

- `run_dir=/root/autodl-tmp/outputs/nano-grpo-qwen05b/<sft_run_name>`
- `final_adapter=<run_dir>/final_adapter`

## 第 2 步：评估纯 SFT adapter

这一步继续复用现有的 `smoke_eval.py`：

```bash
CUDA_VISIBLE_DEVICES=0 python smoke_eval.py \
  --dataset_root /root/autodl-tmp \
  --output_dir /root/autodl-tmp/outputs/nano-grpo-qwen05b/<sft_run_name> \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path /root/autodl-tmp/outputs/nano-grpo-qwen05b/<sft_run_name>/final_adapter \
  --eval_size 128 \
  --offline true
```

这里衡量的是：

- `base model`
- `base model + SFT adapter`

如果 `SFT` 已经明显超过当前最好 GRPO 基线，那说明这条链路非常值得继续。

## 第 3 步：合并 SFT adapter

如果用了 `prepare_sft_to_grpo.sh`，这一步已经自动完成。手动方式如下：

```bash
CUDA_VISIBLE_DEVICES=0 python merge_lora_adapter.py \
  --dataset_root /root/autodl-tmp \
  --run_dir /root/autodl-tmp/outputs/nano-grpo-qwen05b/<sft_run_name> \
  --offline true
```

默认会生成：

- `merged_model=/root/autodl-tmp/outputs/nano-grpo-qwen05b/<sft_run_name>/merged_model`

这个目录就是之后 `GRPO` 的新 base model。

## 第 4 步：启动 vLLM（基于 merged model）

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_HOME=/root/autodl-tmp/hf \
trl vllm-serve \
  --model /root/autodl-tmp/outputs/nano-grpo-qwen05b/<sft_run_name>/merged_model \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1
```

这里直接用本地 `merged_model` 目录，不需要再访问 Hugging Face Hub。

## 第 5 步：从 SFT merged model 继续 GRPO

第一轮 `SFT -> GRPO` 建议更保守一些：

- `reward_scheme=dense_v2`
- `train[:1000]`
- `max_steps=120`
- `learning_rate=2e-6`
- `max_completion_length=160`
- `save_steps=10`

命令如下：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file accelerate_config.yaml \
  train_grpo_gsm8k.py \
  --dataset_root /root/autodl-tmp \
  --model_name /root/autodl-tmp/outputs/nano-grpo-qwen05b/<sft_run_name>/merged_model \
  --train_slice 'train[:1000]' \
  --eval_size 64 \
  --reward_scheme dense_v2 \
  --max_steps 120 \
  --max_completion_length 160 \
  --learning_rate 2e-6 \
  --save_steps 10 \
  --save_total_limit 20 \
  --offline true \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port 8000 \
  --wandb_project nano-grpo-qwen05b \
  --wandb_run_name grpo-gsm8k-qwen25-05b-sftinit-1000
```

## 第 6 步：评估 SFT -> GRPO

这一步评估的是：

- `merged_model`
- `merged_model + GRPO adapter`

例如评估某个 GRPO checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 python smoke_eval.py \
  --dataset_root /root/autodl-tmp \
  --output_dir /root/autodl-tmp/outputs/nano-grpo-qwen05b/<grpo_run_name> \
  --model_name /root/autodl-tmp/outputs/nano-grpo-qwen05b/<sft_run_name>/merged_model \
  --adapter_path /root/autodl-tmp/outputs/nano-grpo-qwen05b/<grpo_run_name>/checkpoint-110 \
  --eval_size 128 \
  --offline true
```

## 推荐比较方式

建议至少比较这三条线：

- 纯 base
- 纯 SFT：`base + SFT final_adapter`
- `SFT -> GRPO`：`merged_model + best GRPO checkpoint`

这样我们能分清：

- 改善到底主要来自 `SFT`
- 还是 `SFT` 之后 `GRPO` 还能继续往上推一截

## 已验证状态

这条链路当前已经做过三段真实 smoke 验证：

- `train_sft_gsm8k.py` 的 1-step 训练可成功完成、保存 adapter、退出 cleanly
- `merge_lora_adapter.py` 可把 `SFT final_adapter` 合并成独立 `merged_model` 目录
- `train_grpo_gsm8k.py --model_name <merged_model_dir>` 可基于 merged SFT model 成功完成 1-step GRPO 预检

也就是说，现在这条链路不是停留在设计稿上，而是已经能实际开跑。
