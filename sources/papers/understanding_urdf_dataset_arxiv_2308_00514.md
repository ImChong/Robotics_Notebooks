# understanding_urdf_dataset_arxiv_2308_00514

> 来源归档（ingest）

- **标题：** Understanding URDF: A Dataset and Analysis
- **类型：** paper
- **来源：** [arXiv:2308.00514](https://arxiv.org/abs/2308.00514)；IEEE RA-L 2024, 9(5):4479–4486；DOI [10.1109/LRA.2024.3381482](https://ieeexplore.ieee.org/abstract/document/10478618)
- **作者：** Daniella Tola（奥尔胡斯大学 ECE）、Peter Corke（昆士兰科技大学 QUT Centre for Robotics）；Innovation Foundation Denmark / MADE FAST 资助
- **代码 / 数据：** https://github.com/Daniella1/urdf_files_dataset（MIT）
- **入库日期：** 2026-08-17
- **一句话说明：** 给出据称首份带元数据的公开 URDF Bundle 语料（322 文件 / 195 独特机型），并分析 xacro 生成、网格类型、跨源重复、ROS 解析失败与多 parser 不一致。

## 核心论文摘录（MVP）

### 1) 规模与意图（Abstract）

- **链接：** <https://arxiv.org/abs/2308.00514>
- **核心贡献：** 322 份 URDF、其中 195 个独特机型；多余者为跨源重复或同机变体。目标是给「URDF 在野外如何被写/被用」打地基，并公开数据集、分析脚本，供工具评测。
- **对 wiki 的映射：**
  - [URDF Files Dataset](../../wiki/entities/urdf-files-dataset.md)
  - [URDF](../../wiki/concepts/urdf-robot-description.md)

### 2) 六源构成（Table II / Sec. III-B）

- **链接：** 论文 HTML Sec. III
- **核心贡献：** ros-industrial 108、random 67、matlab 52、robotics-toolbox 44、oems 35、drake 16。ros-industrial 约占 34%，后续结构/网格统计有来源偏差。变体示例：iiwa 多种 collision、Atlas convex hull vs minimal contact。
- **对 wiki 的映射：**
  - [URDF Files Dataset](../../wiki/entities/urdf-files-dataset.md)
  - [机器人描述目录选型](../../wiki/comparisons/robot-description-catalogs.md)

### 3) 生成、解析与网格（Sec. IV）

- **链接：** 论文 HTML Sec. IV
- **核心贡献：** ~95% 经 xacro；`urdfdom` 3.1.0 下 **11/322** 失败；STL 为最常见视觉/碰撞网格；视觉 Bundle 数（341）高于碰撞（278）。60 机型跨源多重定义（130 Bundle）。
- **对 wiki 的映射：**
  - [URDF](../../wiki/concepts/urdf-robot-description.md)
  - [URDF-Studio](../../wiki/entities/urdf-studio.md)

### 4) 「并不统一」的格式（结论 / 多 parser 表）

- **链接：** 论文 HTML Sec. V–VI
- **核心贡献：** 不同 URDF parser 对同一文件的接受度不一致，说明名称里的 Unified 在实现层并未统一；语料可用于 parser / 转换工具回归，而不是当作最新官方模型源。
- **对 wiki 的映射：**
  - [URDD](../../wiki/entities/paper-urdd-universal-robot-description-directory.md)
  - [robot_descriptions.py](../../wiki/entities/robot-descriptions-py.md)
