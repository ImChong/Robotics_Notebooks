---
type: entity
tags: [infrastructure, gpu-cloud, training, international, notebook, google]
status: complete
updated: 2026-07-02
related:
  - ./google-cloud-gpu.md
  - ./runpod.md
  - ../comparisons/international-gpu-cloud-platforms.md
  - ./mujoco-mjx.md
  - ./brax.md
sources:
  - ../../sources/sites/google-colab.md
summary: "Google Colab 是浏览器内 Jupyter GPU 环境：免费档可用有限 GPU，Pro/Pro+ 提供优先高端卡与更长会话；本库多项目 Colab 教程的默认托管算力入口。"
---

# Google Colab

**Google Colab**（[colab.research.google.com](https://colab.research.google.com/)）是 Google 提供的 **云端 Jupyter Notebook**，一键切换 GPU/TPU runtime，无需自建 VM。

## 一句话定义

在浏览器里 `pip install` + 跑 notebook——最适合 **算法验证、课程作业与 Colab 徽章教程**，而不是多机数周生产训练。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GPU | Graphics Processing Unit | 可选 runtime |
| TPU | Tensor Processing Unit | Google 专用加速器 runtime |
| JAX | JAX | 常与 TPU runtime 搭配 |
| RL | Reinforcement Learning | 小规模实验可行 |
| VM | Virtual Machine | Colab 非完整自控 VM |
| API | Application Programming Interface | 可接 Drive/GCS |

## 为什么重要

- **本库深度嵌入**：MuJoCo MJX、Brax、TSIL、RWM-Lite、dm_control 等均提供 Colab 入口。
- **极低上手成本**：无 SSH、防火墙、安全组。
- **Pro 性价比**：月费订阅换优先 V100/A100 与更长会话，适合反复试错。

## 核心结构 / 机制

| 档位 | 要点 |
|------|------|
| **免费** | 随机 GPU、会话短、空闲易断 |
| **Colab Pro / Pro+** | 优先高端 GPU、~24h 会话、更大内存/盘 |
| **存储** | Google Drive 挂载；大文件走 Drive/GCS |
| **限制** | 通常单卡；无多机分布式 |

## 常见误区或局限

- **不是裸机**：无法任意改内核/驱动；复杂 Isaac Sim 安装常失败。
- **会话会断**：长跑须定期保存到 Drive。
- **多卡训练**：请迁移 [RunPod](./runpod.md) / [Lambda Cloud](./lambda-cloud.md)。

## 与其他页面的关系

- [国外 GPU 云平台选型](../comparisons/international-gpu-cloud-platforms.md)
- [Google Cloud GPU](./google-cloud-gpu.md) — 完整 VM/TPU/Vertex 路径
- [MuJoCo MJX](./mujoco-mjx.md)、[Brax](./brax.md) — 带 Colab 教程

## 推荐继续阅读

- [Colab 首页](https://colab.research.google.com/)
- [Colab Pro 订阅](https://colab.research.google.com/signup)

## 参考来源

- [Google Colab 归档](../../sources/sites/google-colab.md)
