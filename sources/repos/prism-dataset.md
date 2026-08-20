# PRISM（Pressure and Inertial Sensing for Human Motion and Interaction）

> 来源归档

- **标题：** PRISM Dataset
- **类型：** repo / dataset
- **来源：** GRIP 论文配套（Ryosuke Hori 等，CMU / Keio）
- **链接：** <https://github.com/RyosukeHori/PRISM>
- **论文 / 方法：** [GRIP arXiv:2603.16233](https://arxiv.org/abs/2603.16233)；[GRIP 代码](https://github.com/RyosukeHori/GRIP)
- **项目页：** <https://ryosukehori.github.io/grip-project/>
- **入库日期：** 2026-08-20
- **一句话说明：** 同步 **IMU + 鞋垫压力 + 光学 MoCap + 环境模型** 的多模态人体运动数据集；1275 条序列覆盖日常、运动与人–物交互，供 GRIP 训练与评测。

---

## 数据集速查

| 项 | 内容 |
|----|------|
| 被试 | 6（4 男 2 女） |
| 序列 | 1,275 × 10 s（~3.5 h） |
| 采样率 | 100 Hz |
| 标签 | SMPL 姿态（MoSh 拟合 MoCap marker） |
| 传感器 | 腕部/鞋垫 IMU；额外膝/骨盆/头 IMU（采集用）；16 压力 cell / 脚 |
| 场景 | 平地 + 物体交互（踩台、坐物等） |

---

## 与 GRIP 代码的关系

- **快速复现 GRIP：** 使用 [GRIP](https://github.com/RyosukeHori/GRIP) README 中的 **预处理张量**（Google Form），无需重跑原始 pipeline。
- **从原始数据重建：** 下载 PRISM 原始 capture → `GRIP/data_process/kinematics_dataset.py` + `dynamics_dataset.py`。

---

## 对 wiki 的映射

- [paper-grip](../../wiki/entities/paper-grip.md)
- [sources/papers/grip_arxiv_2603_16233.md](../papers/grip_arxiv_2603_16233.md)
