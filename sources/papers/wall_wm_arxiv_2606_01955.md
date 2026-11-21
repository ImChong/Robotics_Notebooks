# WALL-WM: Carving World Action Modeling at the Event Joints（arXiv:2606.01955）

> 来源归档（paper）

- **标题：** WALL-WM: Carving World Action Modeling at the Event Joints
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.01955>（v1 2026-06-01；v2 2026-09-06，cs.RO，页面日期 September 9, 2026）
- **PDF：** <https://arxiv.org/pdf/2606.01955>
- **项目页：** <https://x2robot.com/pages/wm>
- **代码：** <https://github.com/X-Square-Robot/wall-wm>（Apache-2.0）
- **作者：** Shalfun Li\*†, Victor Yao\*, Charles Yang\*, Truth Qu\*, Regis Cheng\*, Ryan Yu\*, Howard Lu\*, Newton Von\*, Vincent Chen\* 等（X Square Robot Team，通讯 Hao Wang‡，Qian Wang）
- **入库日期：** 2026-11-21
- **核对范围：** 通读 `https://arxiv.org/pdf/2606.01955`（v2）正文，含摘要、§1 引言、§2 相关工作、§3 架构（Multi-View Video DiT、Camera RoPE、Sight-cone/Tube Mask、Action DiT 单向层耦合、VLM + Staircase Decoding、五阶段训练）、§7 实验（WorldArena 200 ID + 50 OOD、CO3Dv2 3D 探针、真机四套件 Diverse / Reasoning / Dexterous / Generalization 与 VI-SA 消融）及 §8 讨论与局限；未运行真机实验
- **通读：** 已完成

## 核心摘录（通读 arXiv:2606.01955v2）

- **核心论点（§1）：** 传统定长 chunk 强行把语言（事件级）、视觉（连续动力学）与动作（控制频率）压进同一时钟窗口，导致短程相关性拟合与动作捷径覆盖视觉语义先验。WALL-WM 将学习与数据原子单元改为 **动作语义事件**（reach / grasp / lift / place 等），保证几何结构、T2V 先验与可执行因果性。
- **模型架构（§3）：**
  - 多视角视频 DiT（Wan2.2-TI2V-5B 扩展）：保留视图内自注意力，新增零初始化投影的跨视角分支（VI-SA）、无标定推理的 Camera RoPE，以及仅训练期使用的 sight-cone 与 tube patch 几何掩码。
  - 动作 DiT（同深度随机初始化）：逐层单向交叉注意力读取视频塔 K/V 与本体状态 token，不向视频塔回写；采用非对称时间步（视频仅在中等噪声锚点前向一次，动作跑完整去噪日程）。
  - 双推理模式：Event 模式由人/VLM/agent 给出下一事件描述执行变长段；Unified 模式经 Qwen3.5-9B + Staircase Decoding 并行输出隐式 CoT 指导定长 chunk。
- **实验结果（§7）：**
  - 视频生成（Table 2）：Interaction Quality 0.434（vs Wan2.2-5B 0.226）、Instruction Following 0.391、Semantic Alignment 0.886，图像质量 0.503 略低于基线。CO3Dv2 Point Err 0.271、Depth Err 0.132。
  - 真机四套件（Task Progress）：Diverse 75.86、Reasoning 71.60、Dexterous 32.00、Generalization 53.75（优于 $\pi_{0.5}$、DreamZero、LingBot-VA 及 U-Scratch）；Dexterous 与 U-Scratch（31.25）接近。
  - VI-SA + 事件执行联合消融（Table 4）：Reasoning 32.6 → 71.6，Generalization 22.0 → 53.75。
- **局限（§8）：** 自研平台与预训练数据同源带来评估优势，且基线调参投入不完全对等；目前仍依赖密集时间对齐与事件字幕标注，自监督发现事件边界列为未来工作。

## 对 wiki 的映射

- [paper-rcl-2606-01955-wall-wm-carving-world-action-modeling-at-the-eve](../../wiki/entities/paper-rcl-2606-01955-wall-wm-carving-world-action-modeling-at-the-eve.md)
- 代码归档：[wall-wm.md](../repos/wall-wm.md)
- 项目页归档：[x2robot-wall-wm.md](../sites/x2robot-wall-wm.md)
