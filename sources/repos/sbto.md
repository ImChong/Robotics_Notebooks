# Atarilab/sbto

> 来源归档（ingest）

- **标题：** SBTO — Sampling Based Trajectory Optimization（DynaRetarget 官方实现）
- **类型：** repo
- **URL：** <https://github.com/Atarilab/sbto>
- **Homepage：** <https://atarilab.github.io/dynaretarget.io/>
- **入库日期：** 2026-06-17
- **一句话说明：** **DynaRetarget**（arXiv:2602.06827）核心 **SBTO** 优化器的开源实现：MuJoCo 并行 rollout + Hydra 配置 + CEM 采样，默认在 **OmniRetarget G1–物体** NPZ 参考上做动力学精炼，支持切换 box/chair/shelf/cylinder 等 rollout 场景。

## 核心内容（README 归纳）

### 定位

- 论文 **DynaRetarget** 的 **dynamic refinement** 模块；完整管线（IK 前端 + RL 跟踪 + 真机）以论文与项目页为准，本仓聚焦 **SBTO 轨迹优化**。

### 依赖栈

- Python **3.12.11**；**MuJoCo 3.3.7**；numba / scipy / matplotlib / **hydra-core**；opencv（conda-forge）。

### 安装

```bash
git clone https://github.com/Atarilab/sbto.git
cd sbto
conda create -n sbto python=3.12.11
conda activate sbto
pip install --upgrade pip mujoco==3.3.7 numba==0.62.1 scipy==1.16.2 matplotlib==3.10.6 pyyaml==6.0.3 hydra-core==1.3.2 seaborn==0.13.2
conda install -c conda-forge opencv
pip install -e .
```

### 数据

- 默认示例使用 **OmniRetarget** HF 数据集 `robot-object.zip`：
  `https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/resolve/main/robot-object.zip`

### 用法要点

- 入口：`python3 sbto/main.py solver=cem task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz`
- **Hydra** 覆盖 solver、参考路径、rollout 场景（`task/g1/sim/mj_scene@task.sim.mj_scene=small_box|chair|shelf|cylinder` 等）。
- **双场景：** 参考演示场景 vs rollout 精炼场景分离配置（`mj_scene_ref` / `sim/mj_scene`）。
- **自定义参考：** MuJoCo 格式需设 `task.cfg_ref.flip_quat_pos=False`（OmniRetarget 数据 free joint 为 quat-pos 顺序时为 True）。
- **可视化：**high-level：`scripts/visualize_ref.py` 加载参考轨迹。
- **无物体任务：** `task=g1/robot_ref`。

### 与 SPIDER / OmniRetarget 的关系

- **输入：** 论文实验以 [OmniRetarget](../papers/omniretarget_arxiv_2509_26633.md) 运动学重定向后的 G1–box 轨迹为参考。
- **对照基线：** 论文 Table III 与 **SPIDER** SBMPC 对比；SBTO 在长时域 loco-manipulation 上成功率约 **2×**。

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

- [`wiki/methods/dynaretarget-sbto-motion-retargeting.md`](../../wiki/methods/dynaretarget-sbto-motion-retargeting.md)
- [`sources/papers/dynaretarget_arxiv_2602_06827.md`](../papers/dynaretarget_arxiv_2602_06827.md)
- [`sources/sites/dynaretarget-github-io.md`](../sites/dynaretarget-github-io.md)
