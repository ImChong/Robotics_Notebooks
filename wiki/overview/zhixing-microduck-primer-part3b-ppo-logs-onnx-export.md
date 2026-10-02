---
type: overview
tags: [microduck, onnx, ppo, wandb, tutorial, wechat-curator, export]
status: complete
updated: 2026-10-02
related:
  - ./zhixing-microduck-primer-part3a-cloud-gpu-ppo-training.md
  - ./zhixing-microduck-primer-part4-rust-runtime-onnx.md
  - ../entities/pollen-microduck-rl.md
  - ../concepts/reward-design.md
sources:
  - ../../sources/blogs/wechat_zhixing_microduck_primer_part3b_ppo_logs_onnx_2026-10-02.md
summary: "智践行专辑第③（下）：wandb 读懂 PPO 惩罚符号与课程边界，经 scripts/export.py 导出归一化 ONNX，infer_policy 无头验证仿真推理链。"
---

# 具身智能入门③（下）· 读懂 PPO 产出与 ONNX 导出

## 一句话定义

训练结束后用 **wandb 日志** 判断策略是否在学主任务，再用 **`scripts/export.py` 唯一路径** 得到 Runtime 可吃的 `[1,61]→[1,14]` ONNX。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ONNX | Open Neural Network Exchange | 部署策略格式 |
| PPO | Proximal Policy Optimization | Episode 级奖励日志解读对象 |
| wandb | Weights & Biases | 默认实验跟踪 |
| CSV | Comma-Separated Values | `infer_policy --save-csv` 对齐 Rust 用 |
| xvfb | X virtual framebuffer | 无头 GPU 机跑 infer 常用 |

## 为什么重要

- 手写 ONNX 导出会 **漏观测归一化**，viewer 仍「能走」，真机/Rust 才暴雷。
- 日志符号错误时策略会 **刷惩罚项**；见 AGENTS「Episode_Reward ≤ 0」铁律。

## 核心原理

导出图内 baked normalizer；feed-forward 策略无 LSTM 状态口。Recurrent 版需 `model_api: 2`（本系列默认 walk 为 feed-forward）。

## 工程实践

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --checkpoint 199
xvfb-run -a uv run scripts/infer_policy.py --walking output.onnx --new-cmd-obs --save-csv obs.csv
```

### 日志检查清单

- 主任务 reward 项随迭代 **实质上升**（非仅正则项抬总 reward）。
- 课程 stage 边界若指标 **断崖下跌** → 拉长 stage，勿提前加难度。
- 惩罚项加权值 **不应为正**（农场 butt-hop 等）。

## 局限与风险

- `obs.csv` 中 obs[34:48] 记录的是 **本步 action**；与 Rust 对齐需 **上一步 action** 回填（④ 详述）。
- Ctrl+C 才写 CSV；无头下 Q 退出无效。

## 关联页面

- [④ Rust 运行时](./zhixing-microduck-primer-part4-rust-runtime-onnx.md)
- [Reward Design](../concepts/reward-design.md)

## 参考来源

- [wechat_zhixing_microduck_primer_part3b_ppo_logs_onnx_2026-10-02.md](../../sources/blogs/wechat_zhixing_microduck_primer_part3b_ppo_logs_onnx_2026-10-02.md)

## 推荐继续阅读

- [microduck_rl AGENTS.md — Training ops](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md#training-ops--reading-a-run)
