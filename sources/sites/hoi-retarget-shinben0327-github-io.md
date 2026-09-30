# HOI-Retarget 项目页（shinben0327.github.io/hoi-retarget）

> 来源归档（site）

- **标题：** HOI-Retarget: Contact-Centric Retargeting for Human-Object Interaction
- **类型：** project-page
- **URL：** <https://shinben0327.github.io/hoi-retarget/>
- **论文：** [arXiv:2609.34674](../papers/hoi_retarget_arxiv_2609_34674.md)
- **机构：** Robotic Systems Lab, ETH Zürich
- **入库日期：** 2026-09-30
- **代码：** **已开源** — <https://github.com/shinben0327/hoi-retarget>
- **数据集：** **已发布** — <https://huggingface.co/datasets/shinben0327/hoi-retarget>
- **一句话说明：** 接触中心 HOI 重定向 + 6,952 clip 公开数据集 + G1/H2 多尺度增广与双机协作 demo。

## 核查结论（步骤 2.5）

- **已公开：** Paper / Video / **Code** / **Dataset** / 3D viewer 按钮齐全
- **GitHub：** `shinben0327/hoi-retarget`（BSD-3-Clause）
- **HF：** 数据集 + Space 可视化
- **依赖数据：** SMPL-X、InterMimic/OMOMO 等 **许可下载**，不随仓分发（README `DATA.md`）

## 页面要点摘录

- **Headline：** 0.5 cm mean gap（vs 18.3 cm baseline）；4.6× faster；6,952 clips
- **Embodiment：** Unitree G1、H2；OMOMO 同机位对比身高
- **Object-scale：** ×0.25–×1.5 五档，contact target 在物体系自动跟随
- **Monocular：** CARI4D 重建 + 接触修正工具 + refinement
- **Dynamic：** Kinematic 参考 → DynaRetarget SBTO 或 per-clip RL tracker
