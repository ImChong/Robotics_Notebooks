# CO-Calib

> 来源归档

- **标题：** CO-Calib
- **类型：** repo（即将开源）
- **链接：** https://github.com/HKUST-Aerial-Robotics/CO-Calib
- **维护方：** HKUST Aerial Robotics Group（Shaojie Shen 组）
- **入库日期：** 2026-07-16
- **一句话说明：** 多鱼眼相机标定 **plug-in 数据构造框架**：学习型标定板检测 + 误差分析引导帧选择，提升 Kalibr 类 BA 管线的初始化鲁棒性与成功率。
- **对应论文：** arXiv:2607.05777 — *Observation Quality Matters: Robust Multi-Fisheye Calibration via Failure-Oriented Analysis*

---

## 核心定位

CO-Calib **不替换** 现有 fisheye 相机模型或 bundle-adjustment 求解器，而是在标定前构造 **optimization-ready** 的观测序列：稳定内参 anchor、多相机共视约束与覆盖补全帧。与 [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) 等同属 **HKUST-Aerial-Robotics** 生态。

---

## 对 wiki 的映射

- 论文实体：[paper-co-calib-multi-fisheye-calibration.md](../../wiki/entities/paper-co-calib-multi-fisheye-calibration.md)
- 论文摘录：[co_calib_observation_quality_fisheye_arxiv_2607_05777.md](../papers/co_calib_observation_quality_fisheye_arxiv_2607_05777.md)
