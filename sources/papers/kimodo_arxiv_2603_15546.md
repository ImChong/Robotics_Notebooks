# Kimodo: Scaling Controllable Human Motion Generation

> 来源归档（ingest）

- **标题：** Kimodo: Scaling Controllable Human Motion Generation
- **类型：** paper
- **来源：** arXiv
- **原始链接：**
  - <https://arxiv.org/abs/2603.15546>
  - 项目页：<https://research.nvidia.com/labs/sil/projects/kimodo/>
  - 代码：<https://github.com/nv-tlabs/kimodo>
  - 技术报告 PDF：<https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf>
- **机构：** NVIDIA Research（nv-tlabs / SIL）
- **入库日期：** 2026-05-21
- **一句话说明：** 在 700 小时光学动捕上训练可扩展的 **运动学扩散模型**，通过精心设计的运动表示与 **两阶段 root/body 去噪器**，在文本与全身/末端/2D 路径等多类约束下生成高质量人体与人形运动，并系统分析数据与模型规模对质量与控制精度的影响。

## 核心论文摘录（MVP）

### 1) 动机：公开动捕规模限制生成质量与控制

- **链接：** <https://arxiv.org/abs/2603.15546>
- **摘录要点：** 文本与运动学约束为机器人、仿真与娱乐提供直观的人体运动合成接口，但 **公开 mocap 数据集偏小**，限制了生成质量、控制精度与泛化。Kimodo 在 **700 小时** 光学动捕上训练，强调 **可扩展（scaling）** 与 **可控（controllable）** 并重。
- **对 wiki 的映射：**
  - [Kimodo](../../wiki/entities/kimodo.md) — 「为什么重要」与数据规模叙事
  - [Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md) — 人体/人形域扩散生成对照

### 2) 运动表示 + 两阶段去噪 + 约束条件化

- **链接：** <https://research.nvidia.com/labs/sil/projects/kimodo/>
- **摘录要点：**
  - **表示**：平滑 root（模拟动画工具路径绘制）、全局关节旋转/位置，利于稀疏关键帧；减轻 **floating / foot skating** 等常见伪影。
  - **约束**：全身关键帧、稀疏关节位置/旋转、2D waypoints、稠密 2D paths；约束与噪声运动 **同表示**，对对应元素 **覆写** 并拼接 **mask**。
  - **架构**：文本嵌入 + 约束 + 噪声运动 → **两阶段 Transformer**：先 root denoiser 预测全局 root，再转局部表示给 body denoiser；输出拼接为干净运动。
- **对 wiki 的映射：**
  - [Kimodo](../../wiki/entities/kimodo.md) — 「核心结构/机制」与 Mermaid 流程图
  - [MotionBricks](../../wiki/methods/motionbricks.md) — 同生态「生成式意图层」对照（实时潜空间 vs 显式运动扩散）

### 3) 数据、基准与规模实验

- **链接：** <https://research.nvidia.com/labs/sil/projects/kimodo/docs/benchmark/introduction.html>
- **摘录要点：** 训练依赖 [Bones Rigplay](https://bones.studio/ai-datasets/) 大规模工作室动捕；在 BONES-SEED 上构建 **Kimodo Motion Generation Benchmark**（文本对齐、约束跟随、运动质量等指标）；论文分析 **数据集规模** 与 **模型规模** 对性能的影响。评测嵌入模型 **TMR-SOMA-RP-v1**（700h Rigplay 训练）用于 R-precision、FID 等。
- **对 wiki 的映射：**
  - [Kimodo](../../wiki/entities/kimodo.md) — 评测基准与模型变体选型表
  - [ProtoMotions](../../wiki/entities/protomotions.md) — 生成轨迹导入物理策略的官方下游

### 4) 机器人应用：G1 演示数据与 GEAR-SONIC

- **链接：** <https://research.nvidia.com/labs/sil/projects/kimodo/> · <https://nvlabs.github.io/GEAR-SONIC/demo.html>
- **摘录要点：** G1 骨架变体可 **快于遥操作** 地生成人形演示数据；导出 ProtoMotions / MuJoCo 格式训练物理策略；GEAR-SONIC 在线 Demo 集成「Kimodo 生成运动学轨迹 → 仿真跟踪」。
- **对 wiki 的映射：**
  - [SONIC（规模化运动跟踪）](../../wiki/methods/sonic-motion-tracking.md)
  - [GR00T WholeBodyControl](../../wiki/entities/gr00t-wholebodycontrol.md)

## 引用（仓库 / 项目页）

```bibtex
@article{Kimodo2026,
  title={Kimodo: Scaling Controllable Human Motion Generation},
  author={Rempe, Davis and others},
  journal={arXiv:2603.15546},
  year={2026}
}
```
