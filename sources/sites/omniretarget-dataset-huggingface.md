# OmniRetarget_Dataset（Hugging Face）

- **标题:** OmniRetarget Dataset: Humanoid Loco-Manipulation & Scene Interaction
- **类型:** dataset / huggingface
- **链接:** <https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset>
- **论文:** [OmniRetarget（arXiv:2509.26633）](https://arxiv.org/abs/2509.26633)
- **项目页:** <https://omniretarget.github.io/>
- **生成工具:** OmniRetarget（代码见 [`sources/repos/holosoma.md`](../repos/holosoma.md)）
- **收录日期:** 2026-06-08

## 一句话摘要

Amazon FAR 在 Hugging Face 发布的 **Unitree G1 人形 loco-manipulation / 场景交互** 重定向轨迹集（**约 4.0 小时** 已发布子集）；每条轨迹为 `.npz`，含 `qpos` 与 `fps`；因许可 **不包含 LAFAN1 重定向结果**，但提供重定向代码供用户自行生成。

## 子集结构

| 子目录 | 说明 | 源数据 | 时长 |
|--------|------|--------|------|
| `robot-object/` | 搬运物体动作 | OMOMO | 3.0 h |
| `robot-terrain/` | 攀台等复杂地形动态动作 | 自采 MoCap | 0.5 h |
| `robot-object-terrain/` | 物体 + 地形联合交互 | 自采 MoCap | 0.5 h |
| **合计** | | | **4.0 h** |

- `models/`：URDF、SDF、OBJ 等可视化资产（训练加载轨迹 **不必需**）。

## 数据格式

每个 `.npz` 文件含单条轨迹：

| 键 | 形状 / 含义 |
|----|-------------|
| `qpos` | `[T, D]` 系统状态；机器人 36D（浮动基 7D + 关节 29D）+ 可选物体 7D |
| `fps` | 标量帧率（如 30.0） |

**`qpos` 向量结构：**

- 机器人位姿（36D）：浮动基 `[qw, qx, qy, qz, x, y, z]`（7D）+ 关节角（29D）
- 物体位姿（7D，可选）：`[qw, qx, qy, qz, x, y, z]`

## 快速使用（官方 README 摘要）

```bash
git lfs install
git clone https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset
pip install numpy
```

```python
import glob, numpy as np
paths = glob.glob("robot-object/*.npz")
with np.load(paths[0]) as data:
    qpos = data["qpos"]
    fps = float(data["fps"])
```

**可视化（可选）：** 仓库提供 `visualize.py`（Drake + Meshcat）；脚本内设置 `task` 为 `object` / `terrain` / `object-terrain`。

## 许可与局限

- **LAFAN1 未发布：** 页面明确因 licensing 无法发布 LAFAN1 重定向数据；用户需用 holosoma 重定向代码自行处理 LAFAN1。
- **与论文总规模：** 论文/项目页报告重定向总量 **8–9+ 小时**（含 LAFAN1 等）；HF 当前公开子集为 **4.0 h** OMOMO + 自采 MoCap 部分。

## 对 Wiki 的映射

- **wiki/entities/omniretarget-dataset.md**：数据集实体页。
- **wiki/entities/paper-hrl-stack-03-omniretarget.md**：论文方法与下游 RL 上下文。
- **wiki/concepts/motion-retargeting.md**：重定向问题域交叉引用。
