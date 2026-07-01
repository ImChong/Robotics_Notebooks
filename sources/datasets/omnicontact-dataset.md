# OmniContact Dataset（HuggingFace）

> 来源归档

- **标题：** OmniContact Dataset: Contact-Rich Humanoid Object Interaction
- **类型：** dataset
- **链接：** <https://huggingface.co/datasets/lightcone02/OmniContact-Dataset>
- **论文：** [arXiv:2606.26201](https://arxiv.org/abs/2606.26201)
- **项目页：** <https://omnicontact.github.io/>
- **机构：** 诺亦腾机器人（Noitom Robotics）等
- **入库日期：** 2026-07-01
- **访问条件：** 需登录 HuggingFace 并同意联系方式共享条款后方可下载
- **一句话说明：** 接触丰富的人形 loco-manipulation MoCap 与 **Unitree G1 重定向 NPZ** 轨迹：917 条 processed 序列（7.33 GB）+ 原始 BVH/CSV 与 Viser 可视化工具；含二进制四端接触标签与 70/15/15 推荐划分。

---

## 规模与结构

| 子集 | 描述 | 轨迹数 |
|------|------|--------|
| `npz/box/carry_case_1_4/` | 搬箱 | 388 |
| `npz/box/punt_case_1_2/` | 脚踢球式 punt 箱 | 118 |
| `npz/box/push_case_1_3/` | 手推箱 | 211 |
| `npz/soccer/case1_forward/` | 向前踢球 | 30 |
| `npz/soccer/case2_r2l/` | 右→左球交互 | 30 |
| `npz/soccer/case3_l2r/` | 左→右球交互 | 30 |
| `npz/soccer/case4_carry/` | 带球 | 51 |
| `npz/soccer/case5_pickball_kick/` | 捡球再踢 | 59 |
| **合计** | | **917** |

论文 Appendix A 报告完整采集 **1,274** 有效序列 / **22.29 h** / **7.22M** 物体帧（90 Hz 同步）；HF 子集为公开 G1 NPZ + 原始 MoCap。

附加目录：

- `raw_mocap/box/`、`raw_mocap/soccer/` — BVH、CSV、物体 pose、`capture_meta.json`
- `assets/` — G1 URDF 与物体 mesh
- `metadata/splits.csv` — 70/15/15 train/val/test
- `demo/` — 各任务 MoCap / G1 预览视频与 GLB

---

## NPZ 字段

| 键 | 形状 | 说明 |
|----|------|------|
| `fps` | scalar | 帧率 |
| `base_pos_w` | `[T, 3]` | 浮基位置 |
| `base_quat_w` | `[T, 4]` | 浮基朝向（wxyz） |
| `joint_pos` | `[T, 29]` | G1 关节角 |
| `body_pos_w` / `body_quat_w` | `[T, 39, 3/4]` | 刚体位姿 |
| `object_pos_w` / `object_quat_w` | `[T, 3/4]` | 物体 6D |
| `contact_info` | `[T, 4, 1]` | 左踝、右踝、左腕、右腕 二进制接触 |

---

## 快速用法

```python
import numpy as np

path = "npz/box/carry_case_1_4/20260330000166_1_1775007076_with_contact.npz"
with np.load(path, allow_pickle=False) as data:
    joint_pos = data["joint_pos"]
    contact = data["contact_info"]
    fps = float(data["fps"][0])
```

可视化：`visualize_npz_bvh_minimal.py`（Viser 浏览器，NPZ + BVH + 物体 mesh 同屏）。

---

## 与 OMOMO 对照（论文 Table 6）

| 维度 | OmniContact | OMOMO |
|------|-------------|-------|
| 平均序列时长 | **62.98 s** | 5.69 s* |
| 平均物体路径 | **19.76 m** | 2.67 m* |
| 同步频率 | **90 Hz** | 30 Hz |
| 侧重点 | 长时程 loco-manipulation + contact-flow 监督 | 更多序列/物体类别、短窗交互 |

---

## 对 wiki 的映射

- 论文实体：[OmniContact](../../wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md)
- 论文摘录：[omnicontact_arxiv_2606_26201.md](../papers/omnicontact_arxiv_2606_26201.md)
- [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) — BVH→G1 NPZ 管线
