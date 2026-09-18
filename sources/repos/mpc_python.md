# mpc_python

> 来源归档（ingest）

- **标题：** mpc_python — Iterative MPC path tracking with CVXPY
- **类型：** repo
- **链接：** <https://github.com/mcarfagno/mpc_python>
- **Stars：** ~520（2026-09-18）
- **License：** MIT
- **入库日期：** 2026-09-18
- **一句话说明：** 面向从基础控制过渡到实时凸优化的 **CVXPY 迭代线性化 MPC（iMPC）** 教学仓库：严格 QP 框架 + MuJoCo/MuSHR 小车路径跟踪与静态/动态障碍避障 demo，附 Jupyter 推导笔记。
- **沉淀到 wiki：** [`wiki/entities/mpc-python.md`](../../wiki/entities/mpc-python.md)

---

## 开源状态（2026-09-18 GitHub README 核查）

| 产物 | 状态 |
|------|------|
| MPC 实现 + demo | **已开源** MIT |
| MuJoCo 仿真 demo | **已开源**（`mpc_demo_mujoco.py`） |
| 无物理 headless demo | **已开源**（`mpc_demo_nosim.py`） |
| Jupyter 笔记 | **已开源**（模型推导 / iMPC / 障碍 halfplane 约束，3.x 仍 WIP） |
| Nix flake | **已开源**（`nix run .#mujoco-demo` / `#nosim-demo`） |

---

## 技术要点

- **求解栈：** [CVXPY](https://www.cvxpy.org/) 维护 **QP** 形式，通过 **迭代线性化（iMPC）** 处理非线性运动学，而非 CasADi NLP 主路径。
- **车辆模型：** 集成 [prl-mushr/mushr_mujoco_ros](https://github.com/prl-mushr/mushr_mujoco_ros) MuSHR 阿克曼模型；亦提供 dummy car headless 模式。
- **配置入口：** `config/mpc.yaml`（MPC 参数）、`config/simulation.yaml`（demo 公共配置）。
- **核心代码：** `mpc_python/cvxpy_mpc/cvxpy_mpc.py` + `utils.py`。
- **致谢链：** Borrelli MPC 材料、PythonRobotics、MPCC（alexliniger）、rocket-lander、MuSHR。

---

## 典型入口

```bash
# Conda
conda env create -f env.yml && conda activate simulation
python3 mpc_python/mpc_demo_mujoco.py
python3 mpc_python/mpc_demo_nosim.py

# Nix
nix run --impure .#mujoco-demo
nix run .#nosim-demo
```

GUI MuJoCo 需 `nixGL`（flake 文档）；headless 可直接跑。

---

## 对 wiki 的映射

- [mpc-python 实体页](../../wiki/entities/mpc-python.md)
- [Model Predictive Control](../../wiki/methods/model-predictive-control.md)
- [MuSHR](../../wiki/entities/mushr.md)
- [PythonRobotics](../../wiki/entities/python-robotics.md)
