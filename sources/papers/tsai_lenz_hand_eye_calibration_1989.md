# A new technique for fully autonomous and efficient 3D robotics hand/eye calibration（Tsai & Lenz, 1989）

> 来源归档（ingest）

- **标题：** A new technique for fully autonomous and efficient 3D robotics hand/eye calibration
- **作者：** Roger Y. Tsai, Reuven Lenz
- **类型：** paper
- **期刊：** IEEE Transactions on Robotics and Automation，Vol. 5，No. 3，1989-06
- **DOI（Crossref  canonical）：** <https://doi.org/10.1109/70.34770>
- **用户引用 DOI：** <https://doi.org/10.1109/JRA.1989.28719>（Crossref 未解析；以 **10.1109/70.34770** 为准）
- **入库日期：** 2026-09-27
- **一句话说明：** 经典 **手眼标定** 闭式解：从多组机器人运动与视觉测量解 \(AX=XB\)，实现完全自主、高效的 3D hand/eye 外参标定。

## 核心摘录（面向 wiki 编译）

- **问题：** 确定 **相机坐标系** 与 **机器人末端坐标系** 之间固定变换，使视觉测量可用于抓取/对准。
- **方法：** 利用多组姿态下标定特征的运动约束，将 hand-eye 化为 **\(A X = X B\)** 型矩阵方程并闭式求解旋转再求平移（Tsai 1989 技术路线）。
- **意义：** 后续库（OpenCV `CALIB_HAND_EYE_TSAI`）与大量 Eye-in-hand 产线标定的理论源头之一。
- **对 wiki 的映射：** [hand-eye-calibration](../../wiki/concepts/hand-eye-calibration.md)；[opencv-calib3d-hand-eye](../sites/opencv-calib3d-hand-eye.md)

## 当前提炼状态

- [x] DOI 经 Crossref 核实为 10.1109/70.34770
- [x] 与 OpenCV Hand-Eye API 交叉引用
