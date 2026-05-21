# Kimodo — NVIDIA 官方项目页

- **来源**：https://research.nvidia.com/labs/sil/projects/kimodo/
- **类型**：site（项目页 / 技术展示）
- **机构**：NVIDIA Research（SIL）
- **归档日期**：2026-05-21
- **关联仓库**：https://github.com/nv-tlabs/kimodo
- **技术报告**：[kimodo_tech_report.pdf](https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf)
- **在线文档**：https://research.nvidia.com/labs/sil/projects/kimodo/docs

## 一句话说明

**Kimodo** 是在约 **700 小时** 商业友好型光学动捕上训练的 **运动学扩散模型**，通过文本与多种运动学约束生成高质量 3D 人体与人形运动，并作为 NVIDIA Physical AI / 人形运动栈中的「可控轨迹生成」环节。

## 为什么值得保留

- 项目页比 README 更完整呈现 **能力演示分类**（locomotion、object interaction、dancing、stunts、约束 inbetweening 等）与 **NVIDIA 人形生态互操作图**（SOMA、BONES-SEED、ProtoMotions、SOMA Retargeter、GEM、GEAR-SONIC）。
- 明确 **两阶段去噪器** 与 **运动表示** 的设计动机（平滑 root、约束 mask 覆写、root/body 分解以减少漂浮与脚滑）。
- 给出 **机器人数据应用** 叙事：G1 骨架生成演示数据 → 导出 ProtoMotions / MuJoCo → GEAR-SONIC 在线 Demo 跟踪。

## 核心能力（项目页归纳）

| 条件类型 | 示例 |
|----------|------|
| 文本 | 单提示、组合 locomotion、多段 prompt 序列 |
| 全身约束 | 稀疏关键帧、inbetweening、环境交互姿态 |
| 末端约束 | 手/脚位置与旋转（可组合） |
| 根轨迹 | 2D 路点、稠密 2D 路径（SOMA / G1 演示） |

## 架构要点（项目页「Kinematic Motion Diffusion」）

- **显式运动扩散**：对骨架姿态序列去噪，而非隐空间抽象。
- **运动表示**：平滑 root（贴近动画工具画路径）、全局关节旋转/位置，便于稀疏关键帧约束；约束与噪声运动同表示，**覆写**对应维度并拼接 **constraint mask**。
- **两阶段 Transformer 去噪器**：root denoiser 先预测全局 root，再变换为局部表示输入 body denoiser；输出为两阶段拼接。
- **训练数据**：[Bones Rigplay 1](https://bones.studio/ai-datasets/)（>700h 工作室级动捕 + 文本描述）；部分变体在公开 [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)（约 288h）上训练以便公平对比。

## NVIDIA 人形运动生态（项目页互链）

| 组件 | 角色 |
|------|------|
| SOMA Body Model | Kimodo 训练骨架之一 |
| BONES-SEED | 公开动捕 + Kimodo 评测构造 |
| ProtoMotions | 生成轨迹 → 物理策略训练 |
| SOMA Retargeter | G1 训练数据重定向 |
| GEM (GENMO) | 单目视频运动重建（与 Kimodo「生成」互补） |
| GEAR-SONIC | Kimodo 生成 G1 运动学轨迹 → 仿真跟踪策略 |

## 对 wiki 的映射

1. **[Kimodo（实体页）](../../wiki/entities/kimodo.md)** — 深化架构、评测与下游管线
2. **[Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md)** — 约束式全身扩散的工程范例
3. **[ProtoMotions](../../wiki/entities/protomotions.md)** / **[SONIC](../../wiki/methods/sonic-motion-tracking.md)** — 文生运动 → 物理跟踪闭环
