---
type: entity
tags: [infrastructure, gpu-cloud, training, china, research, education]
status: complete
updated: 2026-07-02
related:
  - ./autodl.md
  - ./featurize.md
  - ./gpushare.md
  - ./gpufree.md
  - ./ai-galaxy.md
  - ../comparisons/china-gpu-cloud-platforms.md
  - ./isaac-lab.md
sources:
  - ../../sources/sites/matpool.md
summary: "矩池云（matpool.com）是国内面向高校与科研的 GPU 云算力平台，按分钟计费，算力市场覆盖 3090/4090/A100 等，实例常配 150–800GB 大容量磁盘与 7×24 支持。"
---

# 矩池云（Matpool）

**矩池云**（[matpool.com](https://www.matpool.com/)）是专注人工智能领域的 **GPU 云服务商**，以算力市场租用主机为核心，并提供专有云、私有云与高校 AI 实训等行业方案。

## 一句话定义

在算力市场按分钟租用 GPU 开发机，默认附带较大本地磁盘与预装框架镜像，适合 **高校师生与科研团队** 做模型训练、渲染与中期实验，而不必自建机房。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GPU | Graphics Processing Unit | 租用与计费核心 |
| AI | Artificial Intelligence | 平台主服务领域 |
| RL | Reinforcement Learning | 机器人学习常见租用场景 |
| SSH | Secure Shell | 远程登录与开发 |
| IDC | Internet Data Center | 机房与区域选择 |
| CAE | Computer-Aided Engineering | 汽车行业方案中的仿真场景 |
| HPC | High Performance Computing | 科研团队高算力任务 |

## 为什么重要

- **教育市场渗透**：官网强调 300+ 高校客户，与本库读者（学生/实验室）重叠度高。
- **磁盘容量优势**：相对部分竞品，3090/4090/A100 实例示例磁盘 150GB–800GB，利于本地缓存数据集与 checkpoint。
- **按分钟计费**：短实验控费更细；另有 AI 功能岛按分计费的轻量算力入口。

## 核心结构 / 机制

| 维度 | 要点 |
|------|------|
| **入口** | 算力市场选卡型 → 立即租用 |
| **计费** | 按分钟；活动体验金 |
| **卡型示例** | A2000 12G、RTX 3090/4090 24G、A100 80G |
| **稳定性** | 宣称 99.9% 机器稳定、7×24 技术支持 |
| **扩展** | 专有云/私有云/软硬一体大模型方案（团队级） |

## 常见误区或局限

- **行业方案页不等于个人租卡流程**：自动驾驶仿真等叙述偏 B 端方案，个人用户仍以算力市场为主。
- **分钟计费需自律**：忘记关机仍持续扣费。
- **与容器云路径差异**：产品形态因机型而异，创建前看清是主机租用还是功能岛。

## 与其他页面的关系

- [国内 GPU 云平台选型](../comparisons/china-gpu-cloud-platforms.md) — 与 AutoDL、Featurize 等并列对比
- [AutoDL](./autodl.md)、[Featurize](./featurize.md) — 同类竞品
- [Isaac Lab](./isaac-lab.md) — 常见高算力训练栈

## 推荐继续阅读

- [矩池云官网](https://www.matpool.com/)
- [算力市场](https://www.matpool.com/)

## 参考来源

- [矩池云官方站](../../sources/sites/matpool.md)
