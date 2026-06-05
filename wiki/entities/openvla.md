---
type: entity
tags: [repo, vla, manipulation, open-source, vision-language-action]
status: complete
updated: 2026-05-27
related:
  - ../methods/vla.md
  - ../entities/lerobot.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ../overview/vla-open-source-repro-landscape-2025.md
sources:
  - ../../sources/repos/openvla.md
summary: "OpenVLA 是开源视觉-语言-动作模型：Prismatic VLM 骨干 + 离散动作 token，支持多数据集预训练与 LoRA/OFT 微调，常与 LeRobot 数据栈配合。"
---

# OpenVLA

**OpenVLA**（[openvla/openvla](https://github.com/openvla/openvla)）提供可复现的 **Vision-Language-Action（VLA）** 训练与推理代码，将视觉-语言预训练能力迁移到 **机器人操作**（单臂/桌面操纵为主）。

## 为什么重要

- **开源可微调**：相对闭源 RT 系，社区可基于 Open X-Embodiment 等数据做 **SFT / LoRA / OFT** 实验。
- **与 VLA 方法论对齐**：体现「图像+语言 → 动作 token/连续动作」范式，见 [VLA 方法页](../methods/vla.md)。
- 与 [LeRobot](./lerobot.md) 互补：LeRobot 偏 **数据与部署管道**；OpenVLA 偏 **大模型权重与训练脚本**。

## 核心结构/机制

- **骨干**：Prismatic-7B 等 VLM，融合 SigLIP/DINO 视觉特征与 Llama 类语言模型。
- **动作表示**：将连续控制 **离散化为 token**，便于自回归生成。
- **训练**：多机器人数据集混合预训练；下游可用 **LoRA、OFT** 降低算力门槛。
- **推理**：提供策略服务器与 Hugging Face 权重；真机需自行对接机器人 SDK。

## 常见误区或局限

- **误区：OpenVLA 负责底盘导航** — 默认面向 **操作空间**；移动导航仍常需 [Nav2](./navigation2.md) 等栈。
- **误区：零样本即可工业部署** — 需目标机器人 **微调、标定与安全围栏**。
- **局限**：与 [人形全身控制](./openloong.md) 的关节级 WBC 是不同层级。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |
| LoRA | Low-Rank Adaptation | 低秩增量微调，低成本适配大模型 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| SFT | Supervised Fine-Tuning | 用监督数据将通用模型适配到特定任务分布 |
| SDK | Software Development Kit | 软件开发工具包 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |

## 参考来源

- [sources/repos/openvla.md](../../sources/repos/openvla.md)
- [openvla/openvla](https://github.com/openvla/openvla)
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*

## 关联页面

- [VLA](../methods/vla.md)
- [LeRobot](./lerobot.md)
- [VLA 开源复现景观 2025](../overview/vla-open-source-repro-landscape-2025.md)

## 推荐继续阅读

- [OpenVLA 项目页](https://openvla.github.io/)
