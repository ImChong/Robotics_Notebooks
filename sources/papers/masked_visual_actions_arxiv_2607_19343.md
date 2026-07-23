# Masked Visual Actions for Unified World Modeling（arXiv:2607.19343）

> 来源归档（ingest）

- **标题：** Masked Visual Actions for Unified World Modeling
- **类型：** paper / masked visual actions / video world model / policy evaluation / inverse modeling
- **arXiv：** <https://arxiv.org/abs/2607.19343>（PDF：<https://arxiv.org/pdf/2607.19343.pdf>）
- **项目页：** <https://masked-visual-actions.github.io>
- **代码：** <https://github.com/HadiZayer/masked-visual-actions>
- **权重：** <https://huggingface.co/HadiZayer/masked-visual-actions>
- **作者：** Hadi Alzayer（Stanford / UMD）、Wenlong Huang、Haonan Chen（Stanford / Harvard）、Christopher Luey、Lvmin Zhang、Maneesh Agrawala、Gordon Wetzstein、Li Fei-Fei、Yilun Du（Harvard）、Jiajun Wu（Stanford）、Jia-Bin Huang（UMD）
- **机构：** 斯坦福大学（Stanford）、马里兰大学学院公园分校（University of Maryland, College Park）、哈佛大学（Harvard University）
- **入库日期：** 2026-07-22
- **一句话说明：** 提出 **Masked Visual Actions**：把动作写成视频中任意实体的 **像素空间部分揭示轨迹**；同一检查点既可作 **前向动力学**（揭示机器人 → 预测场景），也可作 **逆向**（揭示物体目标运动 → 合成机器人运动）。仅用约 **15 小时** 掩码数据微调 Wan-Fun-Control 2.2 14B（LoRA），支撑策略评估（RoboCasa 成功率相关 **r=0.982**）、Best-of-N 规划与 IDM 动作抽取。

## 开源状态（项目页 + 仓库核查，2026-07-22）

- **部分开源：** 项目页挂 GitHub；官方仓含 **推理脚本 + DiffSynth LoRA 训练配方 + HF 双专家 LoRA**（Apache-2.0）；**DROID URDF 控制视频渲染工具** README 仍 *coming soon*；项目页本地 `paper.pdf` **404**（以 arXiv 为准）。论文正文写 “will release code, data, and model weights”——截至入库日代码与权重入口已可见，完整数据/渲染管线未齐。

## 摘要级要点

- **瓶颈：** 视频先验丰富，但如何把 **物理操纵动作** 以与视觉空间对齐的方式注入视频模型，同时仍可跨具身泛化。
- **接口：** 掩码 \(M\) 揭示部分实体时空轨迹；前向 \(S=\) 主动实体（机器人），逆向 \(S=\) 被动实体（物体）；训练只见机器人掩码，物体条件 **零样本** 涌现。
- **数据：** DROID（分割 + URDF 渲染）与 RoboCasa 仿真渲染；成败轨迹均用；约 **15 h** 掩码样本；LoRA rank **256**，约 **10k** step / **8×H200** / **~4 天**。
- **对照：** Skeleton / EEF 条件在未见夹爪与双臂具身上易塌；相对 Ctrl-World / Wan-Move / Wan-I2V，DROID 与 BEHAVIOR（未见双臂）上视觉指标更优。
- **下游：** RoboCasa Best-of-N（Diffusion Policy 提案 + Gemini 3.1 Pro 评判）；策略评估 **r=0.982**；真机四任务演示进度分布对齐；逆设定 + IDM 在 CoffeeServeMug 上相对 DP/ACT/SmolVLA 有竞争力。

## 核心论文摘录（MVP）

### 1) 掩码条件与前向 / 逆向统一

- **链接：** §3；Eq. (1)–(5)；Fig. 3
- **摘录要点：** 视频模型隐式刻画实体轨迹联合分布；条件在子集 \(S\) 上等价于掩码完成。前向：条件主动实体预测被动；逆向：条件被动预测主动。训练无显式 agency，推理任意选 \(S\)。
- **对 wiki 的映射：**
  - [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) — 核心接口与统一建模。
  - [Ctrl-World](../../wiki/entities/paper-ctrl-world.md) / [Wan-Move](../../wiki/entities/paper-wan-move.md) — 文中视觉对照基线（现已独立实体页）。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 像素动作条件谱系。

### 2) 分割 + 渲染双轨数据

- **链接：** §4.1；Fig. 4
- **摘录要点：** SegmentAnything（prompt “A robotic arm”）通用但测试难给精确掩码且遮挡泄漏；URDF/mesh 渲染支持任意动作可视化与推理时控制，但需标定且仅限机器人。二者互补；RoboCasa 半透明渲染 + 红夹爪突出动作。
- **对 wiki 的映射：**
  - [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) — 数据工程。
  - [`sources/repos/masked-visual-actions.md`](../repos/masked-visual-actions.md) — 渲染工具待发布边界。

### 3) 基座与高效微调

- **链接：** §4.2
- **摘录要点：** 基座 **Wan-Fun-Control 2.2 14B**；掩码动作编码后条件生成；**LoRA rank 256**，batch 4，8×H200，~10k steps / 4 天；仅 **15 h** 掩码数据。
- **对 wiki 的映射：**
  - [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) — 训练配方。
  - 官方仓 README — DiffSynth 钉定 commit 与双专家 LoRA。

### 4) 跨具身对照与下游应用

- **链接：** §5.1–5.2；Tab. 1；Fig. 5–11
- **摘录要点：** 同骨干微调 Skeleton / EEF 在未见夹爪与双臂上失败或引入训练域机器人；本方法更稳。RoboCasa 规划 Best-of-N；策略评估 **r=0.982**；真机演示进度对齐；逆建模 + IDM 抽动作（CoffeeServeMug：相对 DP **50%** / ACT **80%** / SmolVLA **85%**，Ours **90%** 量级，见图 11）。
- **对 wiki 的映射：**
  - [world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md) — 规划 / 评估沙盒。
  - [world-models-route-01-cascade](../../wiki/overview/world-models-route-01-cascade.md) — 逆 + IDM 级联。
  - [DriftWorld](../../wiki/entities/paper-driftworld.md) / [OSCAR](../../wiki/entities/paper-oscar.md) — 评估相关但条件不同。

## BibTeX（arXiv；项目页仍为占位）

```bibtex
@article{alzayer2026maskedvisualactions,
  title   = {Masked Visual Actions for Unified World Modeling},
  author  = {Alzayer, Hadi and Huang, Wenlong and Chen, Haonan and
             Luey, Christopher and Zhang, Lvmin and Agrawala, Maneesh and
             Wetzstein, Gordon and Fei-Fei, Li and Du, Yilun and
             Wu, Jiajun and Huang, Jia-Bin},
  journal = {arXiv preprint arXiv:2607.19343},
  year    = {2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-masked-visual-actions.md`](../../wiki/entities/paper-masked-visual-actions.md)
- 代码归档：[`sources/repos/masked-visual-actions.md`](../repos/masked-visual-actions.md)
- 项目页：[`sources/sites/masked-visual-actions-github-io.md`](../sites/masked-visual-actions-github-io.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md)、[world-models-route-01-cascade](../../wiki/overview/world-models-route-01-cascade.md)、[DriftWorld](../../wiki/entities/paper-driftworld.md)、[OSCAR](../../wiki/entities/paper-oscar.md)、[robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)
