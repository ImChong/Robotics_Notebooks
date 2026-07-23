# A Generative Motion Rig for Artist-Driven Motion Authoring

> 来源归档（ingest）

- **标题：** A Generative Motion Rig for Artist-Driven Motion Authoring
- **类型：** paper / SIGGRAPH Talks
- **机构：** DisneyResearch\|Studios · ETH Zürich
- **原始链接：**
  - Disney Research 项目页：<https://studios.disneyresearch.com/2026/07/16/a-generative-motion-rig-for-artist-driven-motion-authoring/>（见 [`sources/sites/disney-generative-motion-rig.md`](../sites/disney-generative-motion-rig.md)）
  - PDF：<https://studios.disneyresearch.com/app/uploads/2026/07/A-Generative-Motion-Rig-for-Artist-Driven-Motion-Authoring-Paper.pdf>
  - DOI：<https://doi.org/10.1145/3799818.3812088>
- **venue：** SIGGRAPH Talks ’26（Los Angeles，2026-07-19–23）
- **入库日期：** 2026-07-23
- **一句话说明：** 把 **通用生成式运动模型** 嵌进 **Blender 插件式 Generative Motion Rig（GMR）**：稀疏关键帧 / Neural Motion Curves / 窗口长度 / 噪声采样构成「generative keyframing」，并支持 MoCap 编辑与传统 FK 层混合。

## 核心论文摘录（MVP）

### 1) 目标：把模型研究接到真实 DCC 工作流

- **链接：** <https://doi.org/10.1145/3799818.3812088>
- **摘录要点：** 生成模型已可用少量稀疏柄与姿态采样整段 3D 运动；缺口在于与 **Maya/Blender 动画软件** 与传统关键帧习惯的集成。本文交付 **Generative Motion Rig** 插件与配套工作流、艺术家反馈与未来挑战，而非新 backbone。
- **对 wiki 的映射：**
  - [Generative Motion Rig（Disney）](../../wiki/entities/generative-motion-rig.md) — 工具/工作流实体页
  - [Blender](../../wiki/entities/blender.md) — DCC 宿主与机器人资产链对照
  - [Character Animation vs Robotics](../../wiki/concepts/character-animation-vs-robotics.md) — 艺术家端生成式关键帧切片

### 2) 系统：client–server + ML-Poser + IBMM + NMC

- **链接：** PDF 同上
- **摘录要点：**
  - **Client–server**：GPU 上跑模型，DCC 客户端可替换（Blender/Maya）；约束变更 → 服务器 → 回写运动。
  - **ML-Poser**（ProtoRes 系神经 IK）由稀疏关节补全全身姿态。
  - **ML-Betweener** 默认用 **Implicit Bézier Motion Model (IBMM)**（Vögeli et al. 2025）保证时域稀疏与平滑；框架可换其他 betweener（如 Flexible Motion In-betweening）。
  - 暴露 **armature 姿态柄** 与 **Neural Motion Curves (NMC)**；控制点可关键帧化；分层存储：generative layer ↔ traditional layer，可混合/切换。
- **对 wiki 的映射：**
  - [Generative Motion Rig](../../wiki/entities/generative-motion-rig.md) — 架构与能力表
  - [机器人关键帧与运动编辑工具](../../wiki/entities/robot-motion-keyframe-editors.md) — 机器人侧关键帧编辑对照（URDF/MJCF vs DCC）

### 3) 生成式创作能力与用户测试

- **链接：** PDF 同上
- **摘录要点：** Direct control（稀疏脚/髋约束改步态与动力学）、噪声重采样（位置 vs 朝向分量效果不同）、时间轴滑动约束改速度、运动外推、基于 inpainting 的 MoCap 编辑、与传统 FK/IK **rig switching / synchronize**。Freestyle：专业艺术家两天内做出 ~22s 追逐镜头；Guided：艺术家 vs 非艺术家不同约束策略；Motion Editing：改跳跃距离并接后退步。
- **对 wiki 的映射：**
  - [Generative Motion Rig](../../wiki/entities/generative-motion-rig.md) — 工作流与局限
  - [Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md) — DCC 集成视角补充

### 4) 局限与未来工作（文内自述）

- **摘录要点：** 训练分布外的非物理/强风格动作仍受限；生成层与传统层最优混合仍开放；加新约束时模型稳定性不足会破坏体验；扩展到更复杂 rig / 不同形态学仍是挑战。synchronize 模式下把预测姿态再当条件可能导致 motion **pop**。
- **对 wiki 的映射：**
  - [Generative Motion Rig](../../wiki/entities/generative-motion-rig.md) — 「局限与风险」

## 开源核查（步骤 2.5）

| 项 | 状态（截至 2026-07-23） |
|----|-------------------------|
| Disney Research 页 | 仅提供 **Download Publication PDF**；无 GitHub / HF / Blender add-on 下载链 |
| 代码 / 插件 | **确认未开源**（项目页与 PDF 均未给出可运行仓库） |
| 结论 | 知识库按 **闭源工具论文** 归档；勿写可复现插件安装步骤 |

> **缩写碰撞注意：** 本文 **GMR = Generative Motion Rig**。本仓库另有 [GMR = General Motion Retargeting](../../wiki/methods/motion-retargeting-gmr.md)（人→机器人重定向），二者无关。

## 当前提炼状态

- [x] SIGGRAPH Talks 全文（3 页）能力、架构、用户测试与局限已摘录
- [x] 开源边界已核查（无代码）

## BibTeX

```bibtex
@inproceedings{buhmann2026generative,
  title     = {A Generative Motion Rig for Artist-Driven Motion Authoring},
  author    = {Buhmann, Jakob and Agrawal, Dhruv and Borer, Dominik and
               V{\"o}geli, Luca and Sumner, Robert W. and Guay, Martin},
  booktitle = {Special Interest Group on Computer Graphics and Interactive Techniques
               Conference Talks (SIGGRAPH Talks '26)},
  year      = {2026},
  address   = {Los Angeles, CA, USA},
  publisher = {ACM},
  doi       = {10.1145/3799818.3812088}
}
```
