# Atarilab/sbto

> 来源归档（ingest · 2026-06-17 深化）

- **标题：** SBTO — Sampling Based Trajectory Optimization（DynaRetarget 官方实现）
- **类型：** repo
- **组织：** [Atarilab](https://github.com/Atarilab)
- **URL：** <https://github.com/Atarilab/sbto>
- **Homepage：** <https://atarilab.github.io/dynaretarget.io/>
- **License：** MIT（`pyproject.toml`）
- **Stars：** ~29（2026-06-17，以 GitHub 为准）
- **Language：** Python
- **入库日期：** 2026-06-17
- **一句话说明：** **DynaRetarget**（arXiv:2602.06827）核心 **SBTO** 优化器：Hydra 配置 + MuJoCo 并行 rollout + CEM 采样，把 **OmniRetarget** 等 kinematic G1–物体 NPZ 参考 refinement 为动力学可行轨迹；支持 box/chair/shelf/cylinder 等 rollout 场景切换。
- **沉淀到 wiki：** [`wiki/entities/sbto.md`](../../wiki/entities/sbto.md)、[`wiki/methods/dynaretarget-sbto-motion-retargeting.md`](../../wiki/methods/dynaretarget-sbto-motion-retargeting.md)

---

## 核心定位

**sbto** 仓库发布论文 **DynaRetarget** 的 **dynamic refinement** 模块（增量时域采样式轨迹优化）。完整管线中的 **IK 前端、PPO tracking、真机部署** 以论文与项目页为准；本仓聚焦 **可复现的 SBTO 优化器** 与 **G1 + 物体** 任务配置。

```
sbto/
├── conf/           # Hydra 配置（task / solver / scene）
├── data/           # 数据加载与参考轨迹接口
├── evaluation/     # 精炼结果评估
├── models/         # 机器人/物体 MuJoCo 模型相关
├── run/            # 运行与 rollout 编排
├── sim/            # MuJoCo 仿真封装
├── solvers/        # CEM 等采样优化器
├── tasks/          # G1 robot-object / robot_ref 等任务
├── utils/
├── main.py         # CLI 入口
└── job.py          # 批处理/作业封装

scripts/
└── visualize_ref.py   # 参考轨迹与场景可视化
```

---

## 依赖栈（README + pyproject.toml）

| 包 | 版本（README 锁定示例） |
|----|-------------------------|
| Python | ≥3.12（README 推荐 3.12.11） |
| mujoco | ≥3.3.7 |
| numpy | ≥2.3.3 |
| numba | ≥0.62.1 |
| scipy | ≥1.16.2 |
| hydra-core | ≥1.3.2 |
| matplotlib / seaborn / pandas / pyyaml / opencv-python | 见 pyproject |

可选 dev：`pytest`、`black`、`ruff`、`mypy`、`pre-commit`。

---

## 安装

```bash
git clone https://github.com/Atarilab/sbto.git
cd sbto
conda create -n sbto python=3.12.11
conda activate sbto
pip install --upgrade pip mujoco==3.3.7 numba==0.62.1 scipy==1.16.2 matplotlib==3.10.6 pyyaml==6.0.3 hydra-core==1.3.2 seaborn==0.13.2
conda install -c conda-forge opencv
pip install -e .
```

---

## 数据（OmniRetarget 默认）

```bash
mkdir datasets && cd datasets
wget "https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/resolve/main/robot-object.zip"
unzip robot-object.zip
```

---

## 用法要点

### 运行 SBTO（CEM 默认）

```bash
python3 sbto/main.py \
  solver=cem \
  task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz
```

- 参考加载细项：`sbto/conf/task/g1/cfg_ref/default.yaml`
- **OmniRetarget NPZ：** free joint 为 `[quat, pos]` → 默认 `task.cfg_ref.flip_quat_pos=True`
- **自定义 MuJoCo 参考：** 设 `task.cfg_ref.flip_quat_pos=False`

### 可视化参考

```bash
python3 scripts/visualize_ref.py \
  task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz \
  task.cfg_ref.speedup=2.
```

### 切换 rollout 物体场景

SBTO 维护 **两套场景**：参考演示（`sbto/conf/task/g1/mj_scene_ref`）与 rollout 精炼（`sbto/conf/task/g1/sim/mj_scene`）。

```bash
python3 sbto/main.py \
  solver=cem \
  task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz \
  task/g1/sim/mj_scene@task.sim.mj_scene=small_box   # 亦可 chair / shelf / cylinder
```

- 自定义物体：支持 primitive、`.urdf`、`.obj` mesh；初始位姿需手动对齐。
- **无物体：** `task=g1/robot_ref`

### Hydra 扩展

运行时可通过 CLI 覆盖 `./conf` 下 solver、task、scene；自定义 task/solver 建议先读 conf 树结构。

---

## 与 OmniRetarget / SPIDER 的关系

| 角色 | 说明 |
|------|------|
| **输入** | 论文实验以 [OmniRetarget](../papers/omniretarget_arxiv_2509_26633.md) kinematic G1–box 轨迹为参考 |
| **对照** | 论文 Table III 与 **SPIDER** SBMPC 对比；SBTO 长时域 loco-manipulation refinement 成功率约 **2×** |
| **下游** | 精炼轨迹供 PPO tracking（论文 mjlab 栈；**未**随本仓发布） |

---

## BibTeX（仓库提供）

```bibtex
@article{dhedin2025dynaretarget,
  title     = {DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization},
  author    = {Dhedin, Victor and Taouil, Ilyass and Omar, Shafeef and Yu, Dian and Tao, Kun and Dai, Angela and Khadiv, Majid},
  journal   = {arXiv preprint arXiv:2602.06827},
  year      = {2025}
}
```

## 对 wiki 的映射

- [`wiki/entities/sbto.md`](../../wiki/entities/sbto.md) — 仓库实体页
- [`wiki/methods/dynaretarget-sbto-motion-retargeting.md`](../../wiki/methods/dynaretarget-sbto-motion-retargeting.md) — 算法与管线归纳
- [`sources/papers/dynaretarget_arxiv_2602_06827.md`](../papers/dynaretarget_arxiv_2602_06827.md)
- [`sources/sites/dynaretarget-github-io.md`](../sites/dynaretarget-github-io.md)
