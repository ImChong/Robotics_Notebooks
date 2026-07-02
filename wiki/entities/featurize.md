---
type: entity
tags: [infrastructure, gpu-cloud, training, china, ml-lab]
status: complete
updated: 2026-07-02
related:
  - ./matpool.md
  - ./autodl.md
  - ./gpushare.md
  - ../comparisons/china-gpu-cloud-platforms.md
  - ./isaac-lab.md
sources:
  - ../../sources/sites/featurize.md
summary: "Featurize（featurize.cn）是面向 ML 研究者的在线实验室，预装主流框架，按量租用 3090/4090/PRO 6000，强调超大免费磁盘（最高约 1.2TB）与 30GB 长效云盘。"
---

# Featurize（蒜粒方块）

**Featurize**（[featurize.cn](https://featurize.cn/)）定位 **机器学习在线实验室**：预装 PyTorch 等环境，提供 RTX 3090/4090 与 **PRO 6000（96GB 显存）** 按量实例，并内置常用数据集访问入口。

## 一句话定义

用浏览器或 SSH 进入已配置好的 GPU 实例，在 **超大本地磁盘** 上直接跑实验，适合不想在多个云盘路径间搬数据的 ML/RL 研究者。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GPU | Graphics Processing Unit | 实例核心资源 |
| ML | Machine Learning | 平台主场景 |
| RL | Reinforcement Learning | 机器人策略训练可用同一套实例 |
| IDE | Integrated Development Environment | 集成开发环境 |
| SSH | Secure Shell | 远程连接方式 |
| VRAM | Video Random Access Memory | PRO 6000 96GB 显存卖点 |

## 为什么重要

- **磁盘即卖点**：4090 实例示例 **750GB**、PRO 6000 **1200GB** 免费磁盘，减少 `/hy-tmp` vs 网盘分裂带来的工程摩擦。
- **环境开箱即用**：对比自建 conda + CUDA 组合，适合快速验证想法。
- **96GB 显存档**：PRO 6000 适合极大 parallel env 或大型 checkpoint 微调。

## 核心结构 / 机制

| 卡型 | 参考配置（官网） | 按量起价 |
|------|------------------|----------|
| RTX 3090 | 6 核、60GB 内存、450GB 磁盘 | ¥1.62/h |
| RTX 4090 | 16 核 EPYC、60GB 内存、750GB 磁盘 | ¥1.87/h |
| PRO 6000 | 96GB 显存、128GB 内存、1200GB 磁盘 | ¥6.19/h |

- **长效存储**：30GB 免费云盘（跨实例持久，官网表述）
- **计费**：按量 + 包月；首充满 5 元送 1 小时体验券

## 常见误区或局限

- **大磁盘≠高速网盘**：仍是实例本地盘逻辑，释放实例前须自行备份关键权重。
- **社区体量小于 AutoDL**：遇到问题时可对照 [国内 GPU 云平台选型](../comparisons/china-gpu-cloud-platforms.md) 选备选平台。
- **图形仿真 GUI**：是否支持 Omniverse 远程桌面需实测镜像，不能仅凭大磁盘推断。

## 与其他页面的关系

- [国内 GPU 云平台选型](../comparisons/china-gpu-cloud-platforms.md)
- [矩池云](./matpool.md)、[恒源云](./gpushare.md) — 磁盘与 OSS 策略不同的竞品
- [Isaac Lab](./isaac-lab.md)

## 推荐继续阅读

- [Featurize 官网](https://featurize.cn/)

## 参考来源

- [Featurize 官方站](../../sources/sites/featurize.md)
