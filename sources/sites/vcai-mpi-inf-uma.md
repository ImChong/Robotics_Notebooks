# UMA 项目页（vcai.mpi-inf.mpg.de/projects/UMA）

- **标题：** UMA: Ultra-detailed Human Avatars via Multi-level Surface Alignment
- **类型：** site / project-page
- **URL：** <https://vcai.mpi-inf.mpg.de/projects/UMA/>
- **会议 / 期刊：** ACM Transactions on Graphics 2026（doi:10.1145/3829365）
- **arXiv：** <https://arxiv.org/abs/2506.01802> — 归档见 [`sources/papers/uma_arxiv_2506_01802.md`](../papers/uma_arxiv_2506_01802.md)
- **代码：** <https://github.com/kv2000/UMA> — 归档见 [`sources/repos/uma.md`](../repos/uma.md)
- **数据集：** <https://gvv-assets.mpi-inf.mpg.de/uma/>
- **Demo：** <https://uma4.umaumau.xyz>
- **作者单位：** Max Planck Institute for Informatics（Saarland Informatics Campus）；Saarbrücken Research Center for Visual Computing, Interaction and AI
- **入库日期：** 2026-08-06

## 一句话摘要

MPI-INF VCAI 组官方项目页：展示 **可驱动超精细着装人体 avatar**（骨骼运动 → 几何 + 外观），强调变焦下的纱线级纹理、多级表面对齐管线、**40×6K** 多视角数据集，以及运动重定向 / 纹理编辑 / VR 查看等应用。

## 项目页核查（步骤 2.5 · 2026-08-06）

| 核查项 | 结论 |
|--------|------|
| **Paper** | 绿按钮 → ACM DL `doi/10.1145/3829365`；仓库另链 arXiv:2506.01802 |
| **GitHub Repo** | 绿按钮 → [`kv2000/UMA`](https://github.com/kv2000/UMA) |
| **Dataset** | 绿按钮 → [`gvv-assets.mpi-inf.mpg.de/uma/`](https://gvv-assets.mpi-inf.mpg.de/uma/)（注册后下载） |
| **Demo** | 绿按钮 → [`uma4.umaumau.xyz`](https://uma4.umaumau.xyz) |
| **开放程度** | **部分开源**：数据集 + 推理 + 交互 demo + checkpoint 已发；训练工具 README 仍 TODO |

- **代码：** <https://github.com/kv2000/UMA>
- **数据集：** <https://gvv-assets.mpi-inf.mpg.de/uma/>
- **模型 checkpoint：** Google Drive（README 链；按被试 `state_dict.pth` / `template.pth` / PCA）

## 公开信息要点（项目页归纳）

### Overview

- 输入：骨骼姿态 + 虚拟相机；输出：超精细着装外观与高保真几何。
- 用户可数字变焦检查纹理细节乃至纱线级图案。
- 新数据集：挑战性纹理图案 + 丰富动力学的多视角 6K 录像；适合 VR/MR 近距观察。

### Pipeline（Figure 2）

- 可驱动模板 \(\mathbf{V}_f\) 注入可学习 latent \(\mathbf{z}_f\)（测试 \(\mathbf{z}_0\)）以建模骨骼无法解释的衣物随机性。
- 纹素超分 \(\mathcal{E}_{\mathrm{sr}}\) 致密可驱动 Gaussian 纹理。
- 多级表面对齐：基础 2D 点跟踪器在栅格化图与 GT 图之间得 \(\mathbf{P}_{f,c,i}\)，抬升并多视角聚合为 3D 对应 \(\tilde{\mathbf{P}}_{f,i}\)，监督顶点/纹素级几何。

### Dataset（Table 1）

| Name | Length (Train) | Length (Test) | Cameras | 标注 |
|------|----------------|---------------|---------|------|
| Subject_0..4 | ~11k–17k | ~6k–10k | 40 | Rigged / Masks / GT Meshes / Pose / Hand / SMPL Params |

每被试提供独立测试序列以验证泛化。

### 应用演示区块

- Free Viewpoint Rendering、Detailed Geometry（跨时间同三角剖分对应）、Motion Retargeting、Texture Editing（纹素对齐几何支撑一致纹理编辑）。

### BibTeX

```bibtex
@article{zhu2026uma,
  title={UMA: Ultra-detailed Human Avatars via Multi-level Surface Alignment},
  author={Zhu, Heming and Sun, Guoxing and Theobalt, Christian and Habermann, Marc},
  journal={ACM Transactions on Graphics},
  publisher={Association for Computing Machinery},
  year={2026},
  doi={10.1145/3829365}
}
```

## 对 wiki 的映射

- 论文实体：[paper-uma](../../wiki/entities/paper-uma.md)
- 交叉：[Face Anything](../../wiki/entities/paper-face-anything-4d-face-reconstruction.md)、[SHELLS](../../wiki/entities/paper-shells-layered-surface-sampling.md)、[SMPL-X](../../wiki/concepts/smpl-x.md)、[遥操作](../../wiki/tasks/teleoperation.md)、[人形训练数据管线](../../wiki/queries/humanoid-training-data-pipeline.md)
