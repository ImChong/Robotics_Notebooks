# NVIDIA Brev Documentation — Overview

> 来源归档

- **标题：** NVIDIA Brev Documentation — Getting Started Overview
- **类型：** site / 官方文档
- **URL：** https://docs.nvidia.com/brev/getting-started/overview
- **门户：** https://brev.nvidia.com
- **入库日期：** 2026-09-06
- **一句话说明：** Brev 提供跨云商的预配置 GPU 实例（驱动/CUDA/Python/Docker/Jupyter 就绪），支持 Launchables 一键环境、VS Code Remote SSH、自定义容器与 NVIDIA NIM 部署。
- **沉淀到 wiki：** [`wiki/entities/nvidia-brev.md`](../../wiki/entities/nvidia-brev.md)

---

## 平台定位（2026-09-06 抓取）

- **即时 GPU：** 选 GPU → 启动实例 → 开始开发；预装 NVIDIA 驱动、CUDA、Python、Docker。
- **与 Physical AI 课程对齐：** Isaac / Omniverse / COMPASS 等教程常指向 Brev Launchable 作为无本地 GPU 的入口。

## 核心概念

| 概念 | 说明 |
|------|------|
| **GPU Instance** | 预配置 VM：GPU + Python + CUDA + Docker + Jupyter |
| **Launchable** | 一键可分享环境：硬件 + 软件 + 代码打包为链接 |
| **Environment** | 预装 AI/ML 工具链的镜像层 |
| **Brev CLI** | 终端工作流（见 [brev_cli.md](../repos/brev_cli.md)） |

## 热门指南（文档索引）

| 级别 | 主题 | 约耗时 |
|------|------|--------|
| Beginner | VS Code Remote SSH | ~10 min |
| Beginner | Jupyter Notebooks | ~10 min |
| Intermediate | Custom Docker Containers | ~15 min |
| Advanced | Deploying NVIDIA NIMs | ~20 min |

## 对 wiki 的映射

- [nvidia-brev](../../wiki/entities/nvidia-brev.md)
- [brev-cli 仓库](../repos/brev_cli.md)
- [nvidia-physical-ai-learning](../../wiki/entities/nvidia-physical-ai-learning.md)
- [compass](../../wiki/entities/compass.md)
