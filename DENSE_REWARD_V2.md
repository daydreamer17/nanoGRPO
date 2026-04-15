# Dense Reward V2

这份文档记录 `nanoGRPO` 下一版 reward 设计方案，目标是在不引入复杂 verifier 的前提下，把当前偏稀疏的训练信号变得更 dense、更稳定。

## 背景

当前版本的 reward 定义在 [train_grpo_gsm8k.py](train_grpo_gsm8k.py)：

- `format_reward`: 只要能解析出 `Final answer: <number>`，奖励 `0.2`
- `answer_reward`: 最终数字与 gold 完全一致，奖励 `1.0`

这套设计已经能工作，但训练早期仍然比较稀疏：

- 格式对了但答案错，只有很弱的奖励
- 接近正确答案和完全离谱的错误，经常都拿到接近 `0`
- 模型容易先学会“会说格式”，但算术进步较慢

## 设计目标

- 保留当前简单、可解释、纯程序化的 reward 风格
- 提高早期训练信号密度
- 区分“格式正确但答案错误”和“完全无效输出”
- 区分“接近正确答案”和“偏差很大”
- 不把 reward 重心从“做对题”挪到“刷格式分”

## 总体方案

建议把总 reward 仍然控制在 `1.20` 左右，避免训练动态变化过大。

推荐拆成 5 个 reward：

1. `strict_format_reward`: `0.15`
2. `brevity_reward`: `0.10`
3. `numeric_parse_reward`: `0.05`
4. `proximity_reward`: `0.25`
5. `exact_answer_reward`: `0.65`

满分为 `1.20`。

## 每个 Reward 的含义

### 1. `strict_format_reward`

目的：强化输出接口稳定性。

规则：

- 最后一行必须是唯一的 `Final answer: <number>`
- `Final answer:` 之后不能再跟其他说明
- 满足条件给 `0.15`
- 否则给 `0.0`

这比当前版本更严格，因为它不再只是“能解析出来就行”，而是要求最后一行严格收束。

### 2. `brevity_reward`

目的：抑制冗长回答，减少被截断概率。

规则：

- 推理行数 `<= 3`，给 `0.10`
- 推理行数 `<= 5`，给 `0.05`
- 更长则给 `0.0`

这里的“推理行数”不包含最后那一行 `Final answer: <number>`。

### 3. `numeric_parse_reward`

目的：奖励“最终答案可结构化解析”的输出。

规则：

- 只要最终答案可以稳定解析为数字，给 `0.05`
- 否则给 `0.0`

这个分数很小，只作为轻量 shaping，不主导训练方向。

### 4. `proximity_reward`

目的：把“接近正确”和“完全错误”区分开。

推荐使用分桶，而不是连续函数，这样更稳也更容易解释。

规则：

- 完全正确：`0.25`
- 相对误差 `<= 1%`：`0.20`
- 相对误差 `<= 5%`：`0.12`
- 相对误差 `<= 20%`：`0.05`
- 否则：`0.0`

说明：

- 如果 gold 为 `0`，应改成绝对误差分桶，避免分母问题
- `proximity_reward` 是 partial credit，不应替代 exact reward

### 5. `exact_answer_reward`

目的：维持“最终做对题”仍是最核心目标。

规则：

- 最终答案与 gold 完全一致，给 `0.65`
- 否则给 `0.0`

之所以从原来的 `1.0` 下调到 `0.65`，是因为现在一部分奖励预算分给了更 dense 的 shaping 和 partial credit。

## 与当前版本的核心区别

当前版本：

- 格式对：`0.2`
- 答案对：`1.0`
- 总体偏稀疏

Dense Reward V2：

- 格式、长度、可解析性都有轻量奖励
- 答案接近正确也有部分奖励
- 完全正确仍然拿最高分

## 这是不是 Process Reward

不是。

这版方案更准确地说是：

- `dense outcome reward`
- 加上若干 `shaping reward`

它仍然不检查中间推理步骤是否正确，所以不属于严格意义上的 process reward。

真正的 process reward 往往需要：

- 中间步骤标注
- 规则化步骤验证器
- 或额外的 judge / verifier model

这会显著提高系统复杂度，因此当前阶段不建议直接引入。

## 示例

### 示例 1：格式正确，答案正确

```text
We need 14 boxes.
Final answer: 14
```

可能得分：

- `strict_format_reward = 0.15`
- `brevity_reward = 0.10`
- `numeric_parse_reward = 0.05`
- `proximity_reward = 0.25`
- `exact_answer_reward = 0.65`
- 总分 `= 1.20`

### 示例 2：格式正确，但答案接近

```text
We need 13 boxes.
Final answer: 13
```

如果 gold 是 `14`，可能得分：

- `strict_format_reward = 0.15`
- `brevity_reward = 0.10`
- `numeric_parse_reward = 0.05`
- `proximity_reward = 0.12` 或 `0.20`
- `exact_answer_reward = 0.0`

这种样本不再和“完全胡说”一起拿 `0`。

### 示例 3：答案看起来合理，但格式不合规

```text
I think the answer is fourteen.
```

可能得分：

- `strict_format_reward = 0.0`
- `brevity_reward = 0.05` 或 `0.10`
- `numeric_parse_reward = 0.0`
- `proximity_reward = 0.0`
- `exact_answer_reward = 0.0`

这类样本仍然会被明确区分为低质量输出。

## 实现建议

建议把当前两个 reward 扩成 5 个，并保留 reward 函数完全程序化。

推荐新增的辅助函数放在 [grpo_gsm8k_utils.py](grpo_gsm8k_utils.py)：

- `extract_final_answer_line`
- `has_trailing_text_after_final_answer`
- `count_reasoning_lines`
- `parse_numeric_value`
- `relative_error`

推荐在 [train_grpo_gsm8k.py](train_grpo_gsm8k.py) 中改成：

```python
reward_funcs = [
    strict_format_reward,
    brevity_reward,
    numeric_parse_reward,
    proximity_reward,
    exact_answer_reward,
]
```

## 实验建议

为了明确观察 reward 设计本身的作用，建议下一轮只改 reward，不同时大改其他超参。

推荐实验设置：

- 保持当前第二轮训练超参
- `train[:512]`
- `max_steps=200`
- `save_steps=10`
- `learning_rate=5e-6`
- `max_completion_length=160`

训练后继续做 checkpoint sweep，再看：

- `tuned_exact_match`
- `tuned_format_hits`
- 不同 checkpoint 的稳定性

## 注意事项

- `format + brevity + parse` 的总权重不要太高，建议不超过 `0.30`
- `exact_answer_reward` 仍然必须是最大头
- `proximity_reward` 最好用分桶，避免 reward 对微小数值波动过于敏感
- 最终模型优劣仍然以 held-out exact match 为主，不要用训练期 reward 直接代替效果评估

## 建议的下一步

最推荐的推进顺序：

1. 先把 `Dense Reward V2` 实现到训练脚本
2. 其余训练超参保持不变
3. 跑一轮新的 `200 step`
4. 做 checkpoint sweep
5. 对比 `run_results.jsonl`

这样最容易判断“效果变化是不是 reward 设计带来的”。
