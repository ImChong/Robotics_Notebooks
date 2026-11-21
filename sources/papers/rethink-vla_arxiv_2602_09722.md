# Rethinking Visual-Language-Action Model Scaling: Alignment, Mixture, and Regularization (Rethink_VLA)

> 来源归档（paper / arXiv:2602.09722v1）

- **URL：** <https://arxiv.org/abs/2602.09722>
- **PDF：** <https://arxiv.org/pdf/2602.09722.pdf>
- **项目页：** <https://research.beingbeyond.com/>（代码仓库附预训练权重）
- **代码仓库：** <https://github.com/BeingBeyond/Rethink_VLA>
- **标题：** Rethinking Visual-Language-Action Model Scaling: Alignment, Mixture, and Regularization
- **作者：** Ye Wang, Sipeng Zheng, Hao Luo, Wanpeng Zhang, Haoqi Yuan, Chaoyi Xu, Haiweng Xu, Yicheng Feng, Mingyang Yu, Zhiyu Kang, et al., Zongqing Lu
- **机构：** BeingBeyond（智在无界）、Renmin University of China、Peking University
- **年份：** 2026（arXiv 2026-02）
- **核对范围：** 已获取并通读 `arXiv:2602.09722v1` PDF 抽取全文（Sections 1–6，涵盖受控骨干架构、182.4M 均衡预训练混合集、Grouped Blind Ensemble 双盲真机评测协议及四大消融维度）；未独立复现 240k 步预训练与真机实验。
- **通读：** 已完成

## 摘要与核心贡献

1. **严格受控的 VLA Scaling 基准实验平台**：采用 **InternVL-3.5-2B**（理解专家）+ **0.7B Action Expert MoT**（Flow Matching 连续动作块 $K=15$）架构，构建总计 **182.4M 样本**的均衡跨本体语料（OXE、AgiBot World、RoboMIND、InternData、SO-100），并在真机 Franka FR3 上设计 **Grouped Blind Ensemble** 双盲对照评测。
2. **四大核心实证结论**：
   - **动作空间对齐优于盲目堆数据（EEF-Relative 最强）**：冻结 VLM 骨干时 `EEF-Relative` 在 LIBERO 5-shot 达 **75.1%**（+8.2% vs Abs），RoboCasa 达 **50.0%**（+4.9%）；且相比速度抖动剧烈的 Delta 动作（Franka 真机高频颤振导致 0% 成功），EEF-Relative 在真机端平滑且泛化最好。
   - **Naive 异构数据混合导致负迁移**：纯 OXE 预训练 `D1`（LIBERO 5-shot `77.3%` / RoboCasa `54.7%`）显著优于简单混入异构真机数据 `D2`（`73.8% / 48.8%`）与仿真数据 `D3`（`72.1% / 49.6%`），证明跨传感器/动作空间未对齐时的暴力混训存在剧烈干扰。
   - **Sensory Dropout 正则化提升抗过拟合与真机鲁棒性**：对本体状态及 ViT 视觉特征施加 `p=0.2` Dropout（`state_dropout_prob=0.2, vit_dropout_prob=0.2`），逼迫策略拒绝单一状态捷径。
   - **两阶段冻结 VLM 课程并非必需**：直接全量协同微调（`Stage 2 Only`, LIBERO 10-shot `85.8%`）略优于或持平先冻后放的两阶段方案（`84.5%`）。

## 与知识库关联

- 实体节点：[cn-os-rethink-vla.md](../../wiki/entities/cn-os-rethink-vla.md)
- 代码归档：[rethink_vla.md](../repos/rethink_vla.md)
