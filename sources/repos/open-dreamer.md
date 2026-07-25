# open-dreamer（next-state）

> 来源归档

- **标题：** Open Dreamer
- **类型：** repo
- **来源：** next-state（GitHub 组织）
- **链接：** https://github.com/next-state/open-dreamer
- **项目页：** https://next-state.github.io/open-dreamer/
- **推理仓：** https://github.com/reactor-team/open-dreamer
- **权重（HF）：** https://huggingface.co/reactor-team/open-dreamer
- **上游论文：** Dreamer 4 — [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527)（[项目页](https://danijar.com/project/dreamer4/)）
- **星标（截至 2026-07-25）：** ~94
- **最近推送：** 2026-07-25
- **主要语言：** Python（JAX / Flax NNX）
- **分类：** 世界模型 / Model-Based RL / 可复现训练管线
- **入库日期：** 2026-07-25
- **一句话说明：** Dreamer 4 的开源 JAX/Flax 实时训练管线：因果视频 tokenizer + 动作条件潜动力学，面向 Minecraft/VPT 风格数据，配套博客与浏览器 demo。
- **沉淀到 wiki：** 是 → [`wiki/entities/open-dreamer.md`](../../wiki/entities/open-dreamer.md)
- **项目页归档：** [`sources/sites/open-dreamer.md`](../sites/open-dreamer.md)
- **推理仓归档：** [`sources/repos/reactor-team-open-dreamer.md`](reactor-team-open-dreamer.md)

---

## README 要点（编译自上游）

- **定位：** 「An open, real-time implementation of the Dreamer 4 world-model pipeline in JAX/Flax」；本仓是 **训练管线**，本地 rollout 走 `reactor-team/open-dreamer`。
- **当前支持：**
  - 训练因果视频 tokenizer
  - 将 Minecraft/VPT 风格 MP4 数据集 tokenize 为 latent ArrayRecord
  - 训练动作条件 latent dynamics
  - 生成 rollout 并计算 FVD
- **Roadmap（未完成）：** Full Dreamer 4 Behaviour-Cloning / RL agent training loop
- **依赖：** Python 3.11、`uv`、CUDA 12 兼容 JAX、Minecraft/VPT 风格 ArrayRecord（见 `dreamer/data/README.md`）
- **包名：** `pyproject.toml` 中 `name = "tiny-dreamer-4"`

## 仓库布局（摘录）

```text
dreamer/          # models / training / generation / checkpointing / data / fvd
scripts/
  train_tokenizer.py
  tokenize_minecraft_dataset.py
  train_dynamics.py
  eval_fvd.py
configs/          # tokenizer / tokenize / dynamics / eval_fvd + dataset YAML
site/             # Next.js 项目页、博客与 live demo
```

## 训练工作流（README）

1. 准备原始 MP4 ArrayRecord shards  
2. 训练 tokenizer（短窗口亦可）  
3. 离线 tokenize 全 episode → latent ArrayRecord + `latent_stats.npz`  
4. 把 latent mean/std 写入 `configs/dataset/minecraft_vpt_latent.yaml`  
5. 训练 dynamics（latent + shifted actions）  
6. `eval_fvd.py`：`mode=generate` 出 MP4，`mode=evaluate` 算 FVD  

## 开源状态（项目页 + README 核查，截至 2026-07-25）

| 资产 | 状态 |
|------|------|
| 训练代码 | **已开源** — `next-state/open-dreamer` |
| 推理/rollout 脚本 | **已开源** — `reactor-team/open-dreamer` |
| 检查点 | **已开源** — HF `reactor-team/open-dreamer`（Orbax，含 `250000/dynamics_ema` 等） |
| 项目页 / 博客 / 浏览器 demo | **已发布** — <https://next-state.github.io/open-dreamer/>（实时模型跑在 Reactor 云端） |
| Dreamer 4 完整 BC/RL agent 环 | **待发布** — README Roadmap 未勾选；博客写明 CoinRun 侧 BC/RL 代码未随 Minecraft 发布 |
| SPDX License 文件 | **未在仓库根检出** — GitHub API `license: null`；使用前需自行确认版权与再分发条款 |

## 对 wiki 的映射

- 实体页：[`wiki/entities/open-dreamer.md`](../../wiki/entities/open-dreamer.md)
- 项目页：[`sources/sites/open-dreamer.md`](../sites/open-dreamer.md)
- 推理仓：[`sources/repos/reactor-team-open-dreamer.md`](reactor-team-open-dreamer.md)
- 沙盒路线对照：[`wiki/overview/world-models-route-03-virtual-sandbox.md`](../../wiki/overview/world-models-route-03-virtual-sandbox.md)
- DreamerV3 前代：[`wiki/entities/paper-shenlan-wm-13-dreamerv3.md`](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md)
