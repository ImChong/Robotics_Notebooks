---
type: entity
tags: [infrastructure, gpu-cloud, training, international, docker, serverless]
status: complete
updated: 2026-07-02
related:
  - ./vast-ai.md
  - ./lambda-cloud.md
  - ./google-colab.md
  - ../comparisons/international-gpu-cloud-platforms.md
  - ./isaac-lab.md
sources:
  - ../../sources/sites/runpod.md
summary: "RunPod 是面向 AI/ML 的 GPU 云：Pods 交互容器、Serverless GPU 与 Secure/Community 双 tier；按秒计费、200+ 模板，机器人 RL headless 训练与推理的常见国外入口。"
---

# RunPod

**RunPod**（[runpod.io](https://www.runpod.io/)）提供 **GPU Pods**（Docker 容器）、**Serverless GPU** 与 **Network Volume** 持久存储，是国外个人开发者与小团队跑深度学习实验的高频平台之一。

## 一句话定义

在浏览器或 API 中秒级启动带 GPU 的 Docker Pod，按秒付费；要 SLA 选 Secure Cloud，要低价实验选 Community Cloud，突发推理可走 Serverless。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GPU | Graphics Processing Unit | Pod 核心资源 |
| SLA | Service Level Agreement | Secure Cloud 提供方承诺 |
| API | Application Programming Interface | REST/GraphQL 自动化 |
| RL | Reinforcement Learning | headless 训练常见场景 |
| SSH | Secure Shell | Pod 远程访问 |
| Docker | Docker Container | 实例隔离单元 |

## 为什么重要

- **卡型覆盖广**：消费级 RTX 至数据中心 H100/B200，适配从 mjlab 调试到多卡 PPO。
- **按秒计费 + 无 egress 宣传**：适合间歇性实验与 checkpoint 拉取。
- **模板生态**：200+ 预构建镜像，减少 CUDA/PyTorch 版本对齐时间。

## 核心结构 / 机制

| 产品 | 用途 |
|------|------|
| **GPU Pods** | 交互式训练/开发，SSH + Jupyter |
| **Serverless** | 按请求扩缩的推理/worker |
| **Secure Cloud** | 数据中心级、带 SLA |
| **Community Cloud** | 低价、可能中断 |

## 常见误区或局限

- **Community 不适合生产推理**：无 SLA，可能抢占。
- **≠ Omniverse 工作站**：图形仿真 GUI 需验证模板与显示转发。
- **Instant Cluster 规模**：单机多卡为主；16+ 机集群见 [Lambda Cloud](./lambda-cloud.md)。

## 与其他页面的关系

- [国外 GPU 云平台选型](../comparisons/international-gpu-cloud-platforms.md)
- [Vast.ai](./vast-ai.md) — 更低价、更不稳定的市场
- [Lambda Cloud](./lambda-cloud.md) — 多机 IB 集群

## 推荐继续阅读

- [RunPod 文档](https://docs.runpod.io/)

## 参考来源

- [RunPod 官方资料](../../sources/sites/runpod.md)
