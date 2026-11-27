# Thinking in Text and Images: Interleaved Vision–Language Reasoning Traces for Long-Horizon Robot Manipulation

- **作者：** Jinkun Liu, Haohan Chi, Lingfeng Zhang, Yifan Xie, YuAn Wang, Long Chen, Hangjun Ye, Xiaoshuai Hao, Wenbo Ding
- **机构：** Tsinghua University, Beijing Institute of Technology, Xiaomi Group
- **版本 / 载体：** arXiv:2605.00438v1 [cs.AI]（2026-05-01），HTML 全文（CC BY 4.0）
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2605.00438) / [arXiv 摘要页](https://arxiv.org/abs/2605.00438)
- **项目页 / 代码：** 未确认开源代码（arXiv:2605.00438v1 原文及公开检索线索均未提供独立项目主页或官方代码仓库链接）
- **核对范围：** 通读全文 §1–§5、Table 1–5、Figure 1–5 图注及 Appendix A–D（含伪轨迹标注 Prompt/JSON Schema、Show-o2 1.5B 与 Wan2.1 VAE 架构、AdamW 训练超参、LIBERO / SimplerEnv-WidowX 主实验与消融/压力测试结果、失效模式分析）
- **通读：** 已完成

## 原文核对笔记

- **核心问题与动机（§1–§2）：** 现有 VLA 策略大多将长时程规划隐式压缩在潜状态中，或仅暴露单一模态推理（纯文本 CoT 缺乏空间接触与目标位姿几何约束；纯视觉未来帧预测多为局部视野且缺乏因果语义约束）。本文提出交错视觉-语言推理框架 **IVLR** 与显式全时程表征 **IVLR-Trace**（由文本子目标 $c^{(i)}$ 与视觉关键帧 $k^{(i)}$ 交替构成的序列 $z = [(c^{(1)}, k^{(1)}), \dots, (c^{(N)}, k^{(N)})]$）。
- **模型与执行机制（§3.1–§3.3）：** 基于 **Show-o2 1.5B** 原生多模态 Transformer，视觉端采用 **Wan2.1 VAE** 压缩并通过 **Flow-Matching** 头生成关键帧潜在表示，文本端自回归解码阶段字幕，动作端引入可学习 `[ACT]` token 与轻量 MLP 预测连续动作。策略在 $t=0$ 从初始观测 $o_0$ 与指令 $\mathcal{I}$ 一次性自生成完整轨迹 $z$ 并缓存，随后每个控制步以闭环方式条件化当前观测 $o_t$、原始指令 $\mathcal{I}$ 与缓存轨迹 $z$（无显式阶段指针，依靠隐式注意力对齐）。
- **伪轨迹监督构建（§3.4, Appendix A）：** 使用 Universal Visual Decomposer (**UVD**) 按视觉嵌入进度切分演示阶段，取各段末帧作为关键帧 $k^{(i)}$，再由 **Qwen3-VL** 生成结构化 JSON 标注（`stage_id`, `caption`, `objects`, `gripper_action`, `spatial_relation`, `keyframe_index`），配合格式、幻觉与重复阶段过滤及人工抽检。
- **训练目标与效率（§3.5–§3.6, Appendix B）：** 总损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{text}} + \lambda\mathcal{L}_{\text{flow}} + \gamma\mathcal{L}_{\text{action}}$（action loss 为 L1），训练时加入随机轨迹噪声与动态上下文掩码。16 张 NVIDIA H200 训练（LIBERO 约 40K 步/4 小时，SimplerEnv 约 60K 步/6 小时）；单张 H20 推理时初始生成全轨迹约需 10 秒，缓存后借助 action chunking 以 10 Hz 闭环执行。
- **核心实验数据（§4.1–§4.5）：**
  - **LIBERO（Table 2）：** 平均成功率 **95.5%**（Spatial 97.8%、Object 95.8%、Goal 97.0%、LIBERO-Long **92.4%**），在 LIBERO-Long 上高于 CoT-VLA（69.0%）与 VLA-0（87.6%）；Object 上略低于 VLA-0（97.8%）与 $\pi_0$-FAST（96.8%）。
  - **SimplerEnv-WidowX（Table 3）：** 总体成功率 **59.4%**（高于 SpatialVLA 42.7%、RoboVLMs 37.5%、Octo-Small 29.5%）。
  - **模态消融（Table 4）：** LIBERO-Long 上无轨迹（w/o Trace）仅 37.7%，纯文本轨迹 62.0%，纯视觉轨迹 68.4%，完整交错轨迹 92.4%。
  - **受控压力测试（Table 5）：** 轨迹生成后施加 1 次 2 cm 末端随机扰动，LIBERO 平均从 95.5% 降至 92.1%（Long 92.4% → 88.2%）；推理时掩码 30% 文本或 30% 视觉关键帧分别降至 92.2% 与 90.0%。
- **明确边界与局限（§4.5, §5, Appendix B–C）：** 仅在仿真环境验证，无真机部署；假设 $t=0$ 工作空间静态且全可观测；约 10 s 初始规划延迟尚未引入异步流水线或滑窗重规划；重度遮挡、大幅物体滑移、规划后场景变化及初始不可见目标仍会导致失败；未报告多种子置信区间。

对 wiki 的映射：[Thinking in Text and Images (IVLR)](../../wiki/entities/paper-agentic-2605-00438.md)
