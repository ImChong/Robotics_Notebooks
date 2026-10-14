# GaussiAnimate: Reconstruct and Rig Animatable Categories with Level of Dynamics

> 来源归档（ingest）

- **标题：** GaussiAnimate: Reconstruct and Rig Animatable Categories with Level of Dynamics
- **类型：** paper
- **机构：** 西湖大学 · 南京大学 · 香港大学 · 澳门科技大学
- **作者：** Jiaxin Wang、Dongxin Lyu、Zeyu Cai、Zhiyang Dou、Cheng Lin、Anpei Chen、Yuliang Xiu
- **来源与会议：** arXiv:2604.08547；SIGGRAPH Asia 2026 Conference Track
- **原始链接：**
  - arXiv 摘要：<https://arxiv.org/abs/2604.08547>
  - PDF：<https://arxiv.org/pdf/2604.08547>
  - 项目页：<https://cookmaker.cn/gaussianimate/>（见 [项目页归档](../sites/gaussianimate.md)）
  - 代码：截至 2026-10-15，项目页标注 Code Coming Soon，未列公开代码仓库
- **入库日期：** 2026-10-15
- **一句话说明：** 提出 Skelebones（Scaffold-Skin Rigging System），从动态 4D Gaussian Splatting 或网格序列自动提取可控骨架与表达非刚性变形的自由形态骨骼，再用 Partwise Motion Matching 生成新动作。

## 核心论文摘录（MVP）

### 1) 动机：自动绑定要兼顾可控结构与柔性形变

- **链接：** <https://arxiv.org/abs/2604.08547>
- **摘录要点：** 传统运动学骨架易控制，却难覆盖衣物、软组织等非刚性表面；自由形态骨骼能贴合表面变形，但缺少直观的运动学层级。论文希望从重建得到的时序 4DGS 或网格中提取可复用的角色绑定，而不依赖人工逐帧绑骨。
- **对 wiki 的映射：**
  - [GaussiAnimate](../../wiki/entities/paper-gaussianimate.md) — Skelebones 表示、PartMM 与适用边界
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md) — 角色资产与机器人动作之间的接口边界

### 2) Skelebones：把内层运动学骨架与外层形变骨骼结合

- **链接：** <https://arxiv.org/abs/2604.08547>
- **摘录要点：**
  - **Bones：** 将时间一致的可变形高斯压缩成自由形态 bones，以近似非刚性表面形变。
  - **Skeleton：** 从规范高斯提取 Mean Curvature Skeleton，并按时间细化拓扑与运动学结构，使骨架适配不同类别和运动。
  - **Binding：** 通过非参数 Partwise Motion Matching（PartMM）将骨架与外层 bones 绑定，通过匹配、检索与混合已有片段合成新骨骼运动。
- **对 wiki 的映射：**
  - [GaussiAnimate](../../wiki/entities/paper-gaussianimate.md) — 表示结构及处理步骤
  - [RigMo](../../wiki/entities/rigmo.md) — 与无标注 Gaussian bones / skinning 的相邻路线比较

### 3) 新动作合成与适用对象

- **链接：** <https://arxiv.org/abs/2604.08547>
- **摘录要点：** PartMM 以检索、匹配和混合已有局部运动来驱动外层形变表示，不依赖学习一个专门的 motion predictor。项目页展示了衣着人物、四足、双足、鸟类和服装等可动画对象；论文同时评估 Gaussian 与 mesh 表示。
- **对 wiki 的映射：**
  - [GaussiAnimate](../../wiki/entities/paper-gaussianimate.md) — 新动作驱动、论文报告与图形学/机器人边界
  - [Character Animation vs Robotics](../../wiki/concepts/character-animation-vs-robotics.md) — 渲染角色运动与物理可执行控制的区别

## 开源核查

| 项 | 状态（截至 2026-10-15） |
|----|-------------------------|
| 官方项目页 | 已开放，含摘要、演示、论文与资源链接 |
| 论文 | arXiv 公开 |
| 代码 | **待发布**：项目页显示 Code Coming Soon，暂未提供 GitHub / Hugging Face 仓库 |
| 数据 | 项目页未提供可下载数据集链接 |
| 结论 | 目前可读论文与查看项目演示；尚不能据公开仓库复现实验管线 |

## BibTeX

```bibtex
@article{wang2026gaussianimate,
  title   = {GaussiAnimate: Reconstruct and Rig Animatable Categories with Level of Dynamics},
  author  = {Wang, Jiaxin and Lyu, Dongxin and Cai, Zeyu and Dou, Zhiyang and Lin, Cheng and Chen, Anpei and Xiu, Yuliang},
  journal = {arXiv preprint arXiv:2604.08547},
  year    = {2026}
}
```
