# Next Iteration Plan After SFT -> GRPO (2026-04-15)

## 本轮结论

这条 `SFT -> GRPO` 路线已经被验证为有效。

当前最重要的结果：

- 纯 `SFT` 最好结果：`34 / 128`
- `SFT merged_model` 作为 base 的复评结果：`35 / 128`
- 本轮 `SFT -> GRPO` 最佳 checkpoint：`checkpoint-100 = 38 / 128`

也就是说，这轮 `GRPO` 相比当前 `SFT` 基线带来了小幅但真实的提升：

- 对 `SFT adapter` 基线：`+4`
- 对 `merged_model` 基线：`+3`

这说明：

- `SFT` 负责把题型和格式先学稳
- `GRPO` 继续在这个基础上小步提高最终答案正确率

## 当前最佳模型

这轮先不要用 `final_adapter` 当默认最佳，而是优先用：

- `/root/autodl-tmp/outputs/nano-grpo-qwen05b/grpo-gsm8k-qwen25-05b-sftinit-1000-lr15e6/checkpoint-100`

它是目前这条路线下最好的 adapter。

## 为什么这轮有效

从结果结构看，这轮增益不是因为“补格式”：

- `SFT merged_model` 本身已经有 `116 / 128` 的 format hits
- `checkpoint-100` 变成 `119 / 128`

格式只提升了一点点，但 exact match 从 `35` 到 `38`，说明这轮 `GRPO` 主要是在：

- 保持格式基本不坏
- 小幅提高最终答案质量

这正是我们希望看到的形态。

## 下一轮建议

下一轮不建议大改方向，也不建议现在扩数据规模。

最稳妥的下一步是：

- 继续使用同一个 `SFT merged_model`
- 保持 `train[:1000]`
- 保持 `reward_scheme=dense_v2`
- 进一步把 `GRPO` 调成更保守、更像“精修”的配置

## 推荐配置

- `model_name`: `/root/autodl-tmp/outputs/nano-grpo-qwen05b/sft-gsm8k-qwen25-05b-1000/merged_model`
- `train_slice`: `train[:1000]`
- `reward_scheme`: `dense_v2`
- `max_steps`: `120`
- `learning_rate`: `1e-6`
- `max_completion_length`: `160`
- `save_steps`: `10`
- `save_total_limit`: `20`
- `eval_size`: `64`

## 为什么这么调

### 1. 不扩数据

当前已经确认：

- `SFT -> GRPO` 路线成立
- 但增益是小幅的

所以现在最重要的是先把这条路榨干，而不是同时把变量再增多。

### 2. 更低学习率

上一轮 `1.5e-6` 已经能带来增益，但幅度不大。  
下一轮把学习率再降到 `1e-6`，更适合在强 `SFT` 基线上做细修。

### 3. 稍微延长到 120 step

上一轮最佳点出现在 `checkpoint-100`，而且正好是训练末尾。  
这意味着还值得往后再给一点空间，但不宜一下拉太长。

`120 step` 是比较稳的下一步：

- 能验证最佳点是不是还在后面
- 又不至于直接进入长尾退化

## 推荐训练命令

### 1. 启动 vLLM

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_HOME=/root/autodl-tmp/hf \
trl vllm-serve \
  --model /root/autodl-tmp/outputs/nano-grpo-qwen05b/sft-gsm8k-qwen25-05b-1000/merged_model \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1
```

### 2. 启动下一轮 GRPO

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file accelerate_config.yaml \
  train_grpo_gsm8k.py \
  --dataset_root /root/autodl-tmp \
  --model_name /root/autodl-tmp/outputs/nano-grpo-qwen05b/sft-gsm8k-qwen25-05b-1000/merged_model \
  --train_slice 'train[:1000]' \
  --eval_size 64 \
  --reward_scheme dense_v2 \
  --max_steps 120 \
  --max_completion_length 160 \
  --learning_rate 1e-6 \
  --save_steps 10 \
  --save_total_limit 20 \
  --offline true \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port 8000 \
  --wandb_project nano-grpo-qwen05b \
  --wandb_run_name grpo-gsm8k-qwen25-05b-sftinit-1000-lr10e6
```

## 推荐 sweep 策略

训练结束后，优先评估这些 checkpoint：

- `60`
- `80`
- `100`
- `110`
- `120`

一键命令：

```bash
cd /root/nanoGRPO
RUN_DIR=/root/autodl-tmp/outputs/nano-grpo-qwen05b/grpo-gsm8k-qwen25-05b-sftinit-1000-lr10e6 \
GPU_ID=0 \
PHASE1_EVAL_SIZE=64 \
PHASE2_EVAL_SIZE=128 \
PHASE1_STEPS_STRING="60 80 100 110 120" \
bash run_dense_v2_eval_sweep.sh
```

## 成功标准

下一轮的合理目标不是暴涨，而是稳定超过当前最好结果：

- 当前最好：`38 / 128`

所以：

- 保守成功：`39 ~ 40 / 128`
- 明显成功：`41 ~ 42 / 128`

如果下一轮仍然停在 `38 / 128` 左右，说明：

- 这条链路已经进入收益递减区
- 下一步更值得改 reward 权重，而不是继续简单堆 step

## 如果下一轮没有继续提升

如果 `120 step + 1e-6` 没有超过 `38 / 128`，下一步建议：

- 保持 `SFT` 起点不变
- 继续 `train[:1000]`
- 不再优先改 step
- 转为调整 reward 权重，让 reward 更偏向 outcome

也就是：

- 降低 `strict_format` / `brevity` 权重
- 提高 `proximity` / `exact_answer` 权重

## 当前一句话总结

下一轮最合理的方案不是推倒重来，而是：

- 保持 `SFT -> GRPO` 路线
- 保持 `train[:1000]`
- 用更低学习率和略长一点的训练，把当前 `38 / 128` 再往上推一小步
