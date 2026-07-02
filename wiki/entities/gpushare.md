---
type: entity
tags: [infrastructure, gpu-cloud, training, china, oss]
status: complete
updated: 2026-07-02
related:
  - ./autodl.md
  - ./featurize.md
  - ./matpool.md
  - ../comparisons/china-gpu-cloud-platforms.md
  - ./isaac-lab.md
sources:
  - ../../sources/sites/gpushare.md
summary: "恒源云（gpushare.com）是国内 AI 共享算力平台，支持按量/包周/包月/包年与个人数据 OSS；实例系统盘约 20GB，数据放 /hy-tmp，关机 10 天自动释放。"
---

# 恒源云（GPUShare）

**恒源云**（[gpushare.com](https://gpushare.com/)）是专注 AI 的 **共享 GPU 算力平台**，提供云市场租卡、官方深度学习镜像与自研 **OSS 个人数据空间**，目标是用较低成本完成云端训练。

## 一句话定义

先在不开实例的情况下用 `oss` CLI 把压缩包传到个人数据，再租 GPU 实例从 `/hy-tmp` 拉取数据训练——适合 **预算敏感、数据集较大** 的短期实验。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GPU | Graphics Processing Unit | 云市场租用单元 |
| OSS | Object Storage Service | 恒源云个人数据空间（自研 CLI） |
| RL | Reinforcement Learning | 常见训练场景 |
| SSH | Secure Shell | 实例登录 |
| CLI | Command Line Interface | `oss` 上传下载工具 |
| IDE | Integrated Development Environment | JupyterLab 等 |

## 为什么重要

- **传数省运行费**：OSS 上传可在实例关机时进行，避免为传数据空跑 GPU 计费。
- **计费周期灵活**：包周适合 1–2 周冲刺实验；按量适合调试。
- **行业横评常客**：常与 AutoDL 并列为高性价比平台。

## 核心结构 / 机制

### 存储

| 路径 | 说明 |
|------|------|
| `/` | 系统盘 **~20GB** |
| `/hy-tmp` | 数据目录；同物理机实例共享，容量登录后 `df -hT` 查看 |

### 典型流程

```mermaid
flowchart LR
  OSS["本地 oss upload\n个人数据"]
  MKT["云市场创建实例"]
  DL["/hy-tmp\noss cp 下载"]
  TRAIN["训练 / shutdown"]
  OSS --> MKT --> DL --> TRAIN
```

- **镜像**：官方 TensorFlow / PyTorch / MXNet / Paddle 等
- **释放**：实例停止 **10 天**自动释放（不可恢复）
- **资源配比**：CPU/内存 = 机器总量 × (租用卡数 / 总卡数)

## 常见误区或局限

- **OSS 仅支持压缩包上传**：目录需先打包；大项目要规划分包策略。
- **`/hy-tmp` 共享**：同主机其他实例可见该目录逻辑，敏感数据注意权限与清理。
- **系统盘很小**：勿把 conda 环境外的巨型数据堆在 `/`。

## 与其他页面的关系

- [国内 GPU 云平台选型](../comparisons/china-gpu-cloud-platforms.md)
- [AutoDL](./autodl.md) — 文档生态更成熟的主要竞品
- [Featurize](./featurize.md) — 大磁盘本地方案对照

## 推荐继续阅读

- [恒源云文档首页](https://gpushare.com/docs/)
- [快速开始](https://gpushare.com/docs/getting-started/quickstart/)

## 参考来源

- [恒源云用户文档](../../sources/sites/gpushare.md)
