# WiLoR：野外 3D 手部定位与重建（CVPR 2025）

- **英文题目：** WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild
- **作者：** Rolandos Alexandros Potamias、Jinglei Zhang、Jiankang Deng、Stefanos Zafeiriou
- **出处：** CVPR 2025，12242–12254
- **CVF 官方论文：** https://openaccess.thecvf.com/content/CVPR2025/html/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.html
- **PDF：** https://openaccess.thecvf.com/content/CVPR2025/papers/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.pdf
- **官方项目：** https://rolpotamias.github.io/WiLoR/
- **原有代码归档：** [sources/repos/wilor.md](../repos/wilor.md)
- **归档日期：** 2026-11-12
- **核对范围：** CVF 论文题录、摘要、图 1、引言及方法概述；完整实验与补充材料尚未系统逐节核查。
- **通读：** 未完成

## 关键摘录

- 由实时全卷积手部定位网络与 Transformer 3D 手部重建模块组成，覆盖多手、左右手区分与高保真对齐。
- 论文报告构造超过 200 万张野外手部图像，并无需额外时序模块就可将单帧估计应用到视频；这不等价于保证全局物理接触或不漂移。
- 对 4D-HOF：WiLoR 提供逐帧 MANO 状态与 2D 关节点，后续仍需 MoGe-3/相机参数、物体配准及条件桥联合细化。

## 对 wiki 的映射

- [复用现有 WiLoR 方法实体](../../wiki/methods/wilor.md)（不新增同名重复实体）
- [4D-HOF 论文实体](../../wiki/entities/paper-4d-hof.md)
