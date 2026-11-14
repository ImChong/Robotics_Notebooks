# Ψ₀：人形移动操作通用基础模型 / Open Foundation Model Towards Universal Humanoid Loco-Manipulation

- **资料类型：** paper
- **原文：** https://arxiv.org/html/2603.12263
- **官方项目：** https://psi-lab.ai/Psi0/
- **官方代码：** https://github.com/physical-superintelligence-lab/Psi0
- **机构：** 南加州大学 Physical Superintelligence Lab（主要作者）、NVIDIA、WorldEngine（官网列出的作者单位）；原综述源材料单写 NVIDIA 不准确。
- **论文版本：** arXiv:2603.12263，RSS 2026
- **核对范围：** 官方论文 arXiv HTML 的摘要、III 模型/三阶段训练/RTC/遥操作、IV 真机对照与消融、附录训练和实时部署；官网架构/数据与评测说明；官方 GitHub README 全文（训练、SIMPLE、G1、部署、SONIC 联动）。
- **通读：** 已完成

## 机制与数据

- 829 小时 EgoDex 人类第一视角视频 + 31 小时 Humanoid Everyday（260 类任务）为核心数据源。**先学 VLM 语义视觉先验**，再冻结 VLM 为真实人形数据训练 flow-based MM-DiT 全身动作专家，最后少量目标任务数据微调。
- Qwen3-VL-2B VLM + 约 500M 动作专家构成高层动作生成；下肢 8-DoF 动作由下层 AMO/RL tracking policy 跟踪。不要写成整个神经模型直接按 1kHz 生成电机力矩。
- 策略一次前向约 160ms，论文提出 RTC 处理 chunk 之间的实时衔接，不等于整环推理 1kHz。
- 原文作者报告用 ~800h 人视频与 ~30h 真机数据，在多个任务上超过部分数据规模大十倍以上的基线，成功率增幅约 40%；是指定基准/协议，不应视作通用保证。

## 官方源码入口

- `scripts/data/raw_to_lerobot.py` 进行数据预处理；
- `scripts/train/psi0/finetune-real-psi0.sh` 为真机任务微调入口；
- `scripts/deploy/serve_psi0-rtc.sh` 启动 policy server；
- `real/scripts/deploy_psi0-rtc.sh` 启动真机 RTC 客户端；
- SIMPLE 仿真提供独立的 `serve_psi0_simple` / evaluation 路径；见官方 README。仅静态核对，未运行真机。

## 对 wiki 的映射

- [Ψ₀ 唯一论文详情](../../wiki/entities/paper-loco-manip-161-156-psi0.md)
