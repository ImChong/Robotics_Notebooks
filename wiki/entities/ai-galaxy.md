---
type: entity
tags: [infrastructure, gpu-cloud, training, china, bare-metal, nvlink]
status: complete
updated: 2026-07-02
related:
  - ./autodl.md
  - ./gpufree.md
  - ./matpool.md
  - ../comparisons/china-gpu-cloud-platforms.md
  - ./isaac-lab.md
sources:
  - ../../sources/sites/ai-galaxy.md
summary: "智星云（ai-galaxy.cn）是亘聪科技旗下 GPU 算力平台，提供云主机、NVLink 云容器、裸金属与 AI 工作站；强调物理 GPU 独享与自动驾驶仿真等裸金属场景。"
---

# 智星云（AI Galaxy）

**智星云**（[ai-galaxy.cn](https://ai-galaxy.cn/)）是上海亘聪信息科技有限公司（安诺其集团子公司）运营的 **GPU 算力服务平台**，产品线从按小时云主机延伸到 **裸金属** 与 **NVLink 多卡容器**。

## 一句话定义

除常规定价租 3090/4090/A100 外，还提供 **整机独占裸金属** 与 **8 卡 NVLink 容器**，适合从个人炼丹到 **自动驾驶仿真、分布式大模型** 的分档算力需求。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GPU | Graphics Processing Unit | 平台核心资源 |
| NVLink | NVIDIA NVLink | 多卡高速互联，云容器卖点 |
| HPC | High Performance Computing | 裸金属常见场景 |
| RL | Reinforcement Learning | 训练算力池适用 |
| SSH | Secure Shell | 远程访问 |
| SLA | Service Level Agreement | 企业级稳定性诉求 |
| IDC | Internet Data Center | 区域与合规 |

## 为什么重要

- **产品分层清晰**：轻量实验走云主机；重仿真走裸金属；千亿参数训练走 NVLink 容器。
- **高校与企业双客户**：官网强调高校用户与企业认证、1V1 服务。
- **机器人相关场景**：裸金属条目明确提及 **自动驾驶仿真** 与工业级模拟。

## 核心结构 / 机制

| 产品 | 适用场景 |
|------|----------|
| **云主机** | 常规定价 GPU VM，弹性扩缩 |
| **裸金属** | 整机独占、近零虚拟化损耗；自动驾驶仿真、重 HPC |
| **云容器** | NVLink/NVSwitch 8 卡并行，大模型训练 |
| **AI 工作站** | 1–4 卡定制，本地+云协同 |

官网展示参考价（随活动变动）：RTX 3090 ¥1.10/h、4090 ¥1.65/h、A100 80G ¥6.6/h。

## 常见误区或局限

- **「物理独享」需选对产品线**：云主机仍是虚拟化路径；极致隔离要租裸金属。
- **价格表随活动波动**：下单前以控制台实时价为准。
- **个人轻量实验可能过剩**：简单单卡 RL 不必默认上裸金属/NVLink。

## 与其他页面的关系

- [国内 GPU 云平台选型](../comparisons/china-gpu-cloud-platforms.md)
- [算力自由](./gpufree.md) — 具身仿真 RT 核心提示的另一国内平台
- [Isaac Lab](./isaac-lab.md) — headless 训练 vs GUI 仿真的算力需求

## 推荐继续阅读

- [智星云官网](https://ai-galaxy.cn/)

## 参考来源

- [智星云官方站](../../sources/sites/ai-galaxy.md)
