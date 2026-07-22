# Masked Visual Actions Project Page

> 来源归档

- **标题：** Masked Visual Actions for Unified World Modeling
- **类型：** site / project page
- **URL：** <https://masked-visual-actions.github.io>
- **论文：** <https://arxiv.org/abs/2607.19343>（项目页 Paper 按钮仍指向本地 `paper.pdf` 且 **404**；以 arXiv 为准）
- **代码：** <https://github.com/HadiZayer/masked-visual-actions>
- **权重：** <https://huggingface.co/HadiZayer/masked-visual-actions>
- **入库日期：** 2026-07-22
- **一句话说明：** 官方项目页展示 **Masked Visual Actions**（像素空间掩码轨迹作统一动作接口）：同一视频世界模型在 **前向**（机器人掩码 → 场景响应）与 **逆**（物体掩码 → 机器人运动）两种设定下工作，并演示策略评估、Best-of-N 规划与跨未见具身泛化。

## 开源状态（项目页核查，2026-07-22）

| 项 | 状态 |
|----|------|
| Paper | arXiv **已发布**（2607.19343）；项目页 `paper.pdf` 链接 **404** / BibTeX 仍为 `XXXX.XXXXX` 占位 |
| Code | 已挂链 — [HadiZayer/masked-visual-actions](https://github.com/HadiZayer/masked-visual-actions) |
| Checkpoints | README / 脚本指向 HF `HadiZayer/masked-visual-actions`（双专家 LoRA） |
| 复现范围 | **部分开源**：推理 + DiffSynth LoRA 训练配方可运行；**DROID URDF 渲染工具** README 写 *coming soon* |
| License | 仓库 **Apache-2.0** |

## 页面结构（策展）

- **TL;DR / Overview** — 参考帧 + 掩码视觉动作 → 生成后续视频；前向规划 / 评估，逆向 + IDM 抽动作
- **Inverse Modeling** — 物体掩码 / 人演示 → 机器人运动（训练仅见机器人掩码，零样本涌现）
- **Policy Evaluation** — 毛巾 / 毛绒等真机演示与模型仿真对齐
- **Why Masked Visual Actions?** — 对照 Skeleton / EEF；未见夹爪与双臂具身上更稳
- **Comparison** — vs Ctrl-World / Wan-Move / Wan-I2V（DROID + BEHAVIOR）
- **Failure Cases** — 精细接触、参考帧未见区域伪影

## 对 wiki 的映射

- 论文归档：[`sources/papers/masked_visual_actions_arxiv_2607_19343.md`](../papers/masked_visual_actions_arxiv_2607_19343.md)
- 代码归档：[`sources/repos/masked-visual-actions.md`](../repos/masked-visual-actions.md)
- 沉淀 **[`wiki/entities/paper-masked-visual-actions.md`](../../wiki/entities/paper-masked-visual-actions.md)**
