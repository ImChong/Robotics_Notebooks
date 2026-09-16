# DIDO（arXiv:2609.15570）

> 来源归档（paper）

- **标题：** DIDO: Distilling Interaction-Centric Dynamics into One-Step Denoising for World Action Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.15570>
- **PDF：** <https://arxiv.org/pdf/2609.15570>
- **项目页：** <https://loveju1y.github.io/DIDO/>
- **代码：** <https://github.com/LoveJu1y/DIDO-WAM>
- **入库日期：** 2026-09-16
- **一句话说明：** 把多步 WAM 去噪蒸馏为一步，用交互实体 bbox token + DINOv3 对齐，避免背景保留、接触动态丢失。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-16）

## 核心摘录

分布匹配外，对夹爪/目标物/交互区域做 bbox 推理 token；DINOv3 特征对齐目标物表征。

**文内指标：** LIBERO 99.0%、LIBERO-Plus 76.6%、RoboTwin 92.0%（作者报告）；含真机长程与泛化实验。

## 对 wiki 的映射

- [paper-dido-wam](../../wiki/entities/paper-dido-wam.md)
- [12 篇技术地图](../../wiki/overview/vla-deploy-12-papers-technology-map.md)
