# xiu-cs.github.io/FMPose3D（FMPose3D 项目页）

> 来源归档（ingest）

- **标题：** FMPose3D — Monocular 3D Pose Estimation via Flow Matching
- **类型：** site / project-page
- **官方入口：** <https://xiu-cs.github.io/FMPose3D/>
- **入库日期：** 2026-07-17
- **一句话说明：** CVPR 2026 论文配套站点：展示 **条件 Flow Matching 2D→3D 提升**、训练/推理示意图、人与动物 demo 视频，以及 arXiv / 代码链接。

## 页面公开信息（检索自 2026-07-17）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://xiu-cs.github.io/FMPose3D/> |
| arXiv | <https://arxiv.org/abs/2602.05755> |
| 代码 | <https://github.com/AdaptiveMotorControlLab/FMPose3D> |

## 源码开放核查（步骤 2.5）

- **开放程度：已开源（代码）+ 部分受限（权重）**
  - **训练/推理代码：** GitHub 仓库公开，**Apache-2.0**。
  - **预训练权重：** Hugging Face `MLAdaptiveIntelligence/FMPose3D` 与 Google Drive；README 标明模型 **Non-Commercial（CC BY-NC-ND）**。
  - **PyPI 包：** `pip install fmpose3d`（Python 3.10）。
  - **生态集成：** README 称 **2026-03 已集成 DeepLabCut** 动物 3D 管线。

## 与论文一致的公开主张（便于 wiki 溯源）

1. **Abstract：** 单目 3D 姿态因深度歧义宜采用概率多假设；扩散准确但推理贵；**Flow Matching** 用 ODE 速度场 **少步积分** 生成 3D 样本；**RPEA** 近似 Bayes 后验期望得到单预测。
2. **Overview — Training：** 从 $x_0\sim\mathcal{N}(0,I)$ 与 GT $x_1$ 线性插值得 $x_t$，网络 $v_\theta(x_t,t,c)$ 预测速度，最小化 CFM 损失。
3. **Overview — Inference：** 从噪声 Euler 积分 $S$ 步得到 $\hat{x}^{3D}$；不同噪声种子 → 多样假设。
4. **Demo：** Human / Animal 单目输入与 3D 可视化对比。
5. **BibTeX：** arXiv:2602.05755，CVPR 2026 accepted。

## 对 wiki 的映射

- [`wiki/entities/paper-fmpose3d-monocular-3d-pose-flow-matching.md`](../../wiki/entities/paper-fmpose3d-monocular-3d-pose-flow-matching.md) — 方法栈、开源状态与基准定位
- [`sources/repos/fmpose3d.md`](../repos/fmpose3d.md) — 仓库与 PyPI 侧归档
