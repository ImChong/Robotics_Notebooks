# AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems

> 来源归档（论文原文）

- **标题：** AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2503.06669>（v5, 2026-10-20；HTML：<https://arxiv.org/html/2503.06669>）
- **项目页：** <https://agibot-world.com/>
- **代码：** <https://github.com/OpenDriveLab/AgiBot-World>
- **作者：** Team AgiBot-World（HKU、AgiBot Inc.、Shanghai Innovation Institute、Shanghai AI Lab）
- **许可：** CC BY-NC-SA 4.0
- **入库日期：** 2026-11-21
- **核对范围：** 通读 arXiv HTML 全文（<https://arxiv.org/html/2503.06669>，Abstract 与 §I–§VI，含 AgiBot G1 硬件与数据采集协议、ViLLA 三阶段架构、GO-1 缩放律与数据质量消融）；未运行 GPU 训练或部署服务。
- **通读：** 已完成
- **一句话说明：** 构建包含超 100 万条轨迹（2976.4 小时、217 任务、87 技能、106 场景）的真机双臂操作平台 AgiBot World，并提出分层 ViLLA（Vision-Language-Latent-Action）通才策略 GO-1，在真机复杂与长程任务上显著优于 OXE 预训练与 RDT/π₀ 基线。

## 核心论文要点（通读提炼）

1. **硬件与采集（§III-A–C）：** 基于 100+ 台同构 AgiBot G1（双 7-DoF 臂 + 移动底盘 + 可调腰部 + 模块化夹爪/视触觉/6-DoF 灵巧手 + 8 相机，30 Hz 记录），结合 VR 遥操作与全身动捕；经可行性验证、云端质检、failure recovery（约 1% 失败恢复轨迹）与 human-in-the-loop 迭代去空闲帧。
2. **数据分布（§III-C）：** 覆盖家居、零售、工业、餐饮、办公五大场景与 3,000+ 物体；多数轨迹 30–60s（最长 >2min）并含子步骤关键帧与语言标注。
3. **ViLLA 三阶段策略 GO-1（§IV）：**
   - **Stage 1（LAM）：** 在网络视频（如 Ego4D）与跨本体机器人数据上用 VQ-VAE 潜动作模型（$k=4$ 离散潜动作 token）学习逆动力学与前向帧预测。
   - **Stage 2（Latent Planner）：** 基于 InternVL2.5-2B + 24 层双向注意力 transformer，根据三视角图像与指令预测潜动作 token $z_t$。
   - **Stage 3（Action Expert）：** 扩散动作头以本体状态与潜规划条件化输出 $H=30$ 连续低层动作块（GO-1 Air 不含 Latent Planner）。
4. **主要实验结论（§V）：** 预训练数据从 OXE 换为 AgiBot World，RDT 在可见/OOD 场景均提升约 +0.30/+0.29 完成分；GO-1 较 RDT 提升约 32%，Latent Planner 带来平均 +0.12 分增益；从 9.2k 到 1M 轨迹呈幂律缩放（$r=0.97$）；人工质检筛选数据比未筛选多得 +0.18 分。

## 对 wiki 的映射

- [AgiBot-World Colosseo：数据与通才策略平台](../../wiki/entities/paper-sa-2503-06669-agibot-world-colosseo-a-large-scale-manipulation.md)
- [官方数据与 GO-1 仓库核查](../repos/agibot-world.md)
