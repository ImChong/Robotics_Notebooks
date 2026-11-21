# Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation

> 来源归档（论文原文）

- **标题：** Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2508.05635>（v3, 2025-11-04；HTML：<https://arxiv.org/html/2508.05635>）
- **项目页：** <https://genie-envisioner.github.io/>
- **代码：** <https://github.com/AgibotTech/Genie-Envisioner-V1>
- **作者：** Yue Liao\*, Pengfei Zhou\*, Siyuan Huang\*, Donglin Yang, Shengcong Chen, ..., Shuicheng Yan, Maoqing Yao, Guanghui Ren（AgiBot Genie Team、LV-NUS Lab、BUAA）
- **许可：** CC BY 4.0（论文）；代码仓分别含 Apache-2.0 与 CC BY-NC-SA 4.0
- **入库日期：** 2026-11-21
- **核对范围：** 通读 arXiv HTML 全文（<https://arxiv.org/html/2508.05635>，Abstract 与 §1–§9，含 GE-Base 两阶段预训练、GE-Act 异步慢快推理、跨本体小样本迁移、GE-Sim 分层动作条件仿真、EWMBench 指标与 Limitations）；未运行 GPU 训练或推理。
- **通读：** 已完成
- **一句话说明：** 在单个视频生成世界模型平台内统一感知、策略学习、仿真与评测：GE-Base 在 AgiBot-World-Beta 上预训练多视角稀疏记忆视频 DiT，并分别支撑 GE-Act 异步慢快控制、GE-Sim 动作条件闭环仿真及 EWMBench 三维评测。

## 核心论文要点（通读提炼）

1. **GE-Base（§2）：** 基于 LTX-Video 2B / COSMOS2 2B 多视角 DiT + 稀疏历史记忆帧（头/左/右三视角，交叉视角注意力）；两阶段预训练：Stage I（GE-Base-MR，3–30 Hz 多帧率适应）与 Stage II（GE-Base-LF，5 Hz 低帧率策略对齐）。
2. **GE-Act（§3）：** 平行接在 GE-Base 旁的 160M 自回归流匹配动作解码器；采用慢快异步推理（视频 DiT 5 Hz 单步去噪缓存潜特征，动作头 30 Hz 五步去噪，在机载 RTX 4090 上约 200 ms 生成 54 步轨迹）。支持 1 小时遥操作数据向 Dual Franka、Agilex Cobot Magic 与 RoboTwin 跨本体小样本适配（§4）。
3. **GE-Sim（§5）：** 将 14 维双臂 EEF 轨迹通过 Pose2Image（投影像素 + 深度圆半径 + 轴向/开合编码）与 Motion Vector 注入 GE-Base-MR，实现动作条件闭环视觉仿真。
4. **EWMBench（§6）与局限（§8）：** 覆盖 Scene Consistency、Action Trajectories Quality（SA/TA/DYN）与 Motion Semantics（Qwen2.5-VL-7B + CLIP）；局限包括预训练仅用单平台 AgiBot-World-Beta、仅限平行夹爪上身桌面操作、仍含部分代理指标。

## 对 wiki 的映射

- [Genie Envisioner：世界模型、策略与视觉仿真](../../wiki/entities/paper-sa-2508-05635-genie-envisioner-a-unified-world-foundation-plat.md)
- [Genie Envisioner 官方源码核查](../repos/genie-envisioner-v1.md)
