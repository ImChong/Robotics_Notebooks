# bfm4humanoid.github.io（BFM 项目页）

- **标题：** Behavior Foundation Model for Humanoid Robots — 官方项目页
- **类型：** site / project-page
- **URL：** <https://bfm4humanoid.github.io/>
- **入库日期：** 2026-05-18
- **配套论文：** [BFM（arXiv:2509.13780）](https://arxiv.org/abs/2509.13780) — 归档见 [`sources/papers/bfm_humanoid_arxiv_2509_13780.md`](../papers/bfm_humanoid_arxiv_2509_13780.md)

## 一句话摘要

BFM 论文官方项目页：以 **演示视频** 展示一个统一策略在 Unitree G1 上覆盖 **motion tracking、locomotion、VR 全身遥操作、行为插值（如 Roundhouse Kick）、新技能少样本获取（Side Salto）** 等多种控制接口；代码标注 *In Coming*。

## 公开信息要点（截至入库日）

- **作者与机构**：Weishuai Zeng, Shunlin Lu, Kangning Yin, Xiaojie Niu, Minyue Dai, Jingbo Wang, Jiangmiao Pang；北京大学 / 港中大（深圳）/ 上交大 / 复旦 / **上海人工智能实验室**。
- **代码 / 权重**：页面写 *Code: In Coming*；以官方仓库后续更新为准。
- **演示分组**（按页面展示顺序，归纳）：
  - 通用动作：游泳、行走 / 坐姿、地面起身。
  - 高动态：Roundhouse Kick、Basketball Layup、Forward Roll、Butterfly Kick、Cartwheel、Side Salto。
  - 接口演示：摇杆 locomotion、VR 全身遥操作（含 HybrIK）。

## 为何值得保留

- **行为组合可视化**：项目页通过视频直观展示 **潜空间插值** 与 **mode 组合**（如 root + keypoint 同时激活），便于配合论文 §4 阅读；同时给出 HOVER / Specialist 等基线的对比视频。
- **新技能少样本获取证据**：Side Salto 等论文表外的额外演示，多放在项目页而非正文；归档站点便于后续回看。
- **同期 BFM 工作对照入口**：与 [BFM-Zero](https://lecar-lab.github.io/BFM-Zero/) 等命名相近但方法不同的同期工作放在一起，有助于读者对比理解。

## 站点结构与可引用锚点

- `#abstract` — 摘要段
- 演示视频区无显式 id；正文引用时建议直接命名「项目页 Side Salto 演示」等。

## 关联资料

- 论文归档：[`sources/papers/bfm_humanoid_arxiv_2509_13780.md`](../papers/bfm_humanoid_arxiv_2509_13780.md)
- BFM 综述：Yuan et al., arXiv:2506.20487（[abs](https://arxiv.org/abs/2506.20487)）
- 同名同期方向：[BFM-Zero（无监督 RL + FB 表示）](https://lecar-lab.github.io/BFM-Zero/)
