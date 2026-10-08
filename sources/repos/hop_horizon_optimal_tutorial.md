# rap-lab-org/public_HOP_horizon_optimal_tutorial

> 官方代码仓库来源归档，核验日期：2026-10-08。

- **仓库：** <https://github.com/rap-lab-org/public_HOP_horizon_optimal_tutorial>
- **项目页：** <https://rap-lab.github.io/research/hop>
- **论文：** [HOP: Fast Differential Dynamic Programming for Horizon-Optimal Trajectory Planning（RSS 2026）](../papers/hop_2026_rss_186.md)
- **论文节点：** [HOP](../../wiki/entities/paper-hop-horizon-optimal-trajectory-planning.md)
- **代码状态：** 公开可读、含 runnable Python 示例。
- **仓库性质：** 教程/示例实现，不是机器人平台集成包；仓库文件列表未见 LICENSE 文件，许可需向作者核实。
- **入库日期：** 2026-10-08

## 代码结构

| 文件 | 职责 |
|---|---|
| HOP_colab_notebook.ipynb | notebook 版教程 |
| utils.py、systems.py | double integrator 与 quadrotor toy systems、滚动/数值工具 |
| lqr.py | brute-force 时域枚举、增广系统构造、HOP-LQR horizon search |
| ddp.py | 有限差分线性化、iLQR backward/forward pass、HOP-DDP solve_hop |
| run_double_integrator.py | HOP-LQR 与 brute force 对照 |
| run_quadrotor.py | 12-state quadrotor 的 HOP-DDP 与 brute-force 对照 |
| requirements.txt、environment.yml | 依赖列表；Conda 环境声明 Python 3.13、NumPy、Matplotlib |

## README 快速运行

```bash
conda env create -f environment.yml
conda activate hop
python run_double_integrator.py
python run_quadrotor.py --skip-bruteforce
```

不使用 Conda 时，README 给出的依赖安装方式是 python3 -m pip install -r requirements.txt，随后运行相同脚本。quadrotor 的 brute-force 对照更慢；--skip-bruteforce 只运行 HOP。

## 复现范围

脚本构造 toy dynamics 与目标、运行离线轨迹优化，报告最佳离散时域、cost、耗时并可保存图像。仓库没有真实机器人传感/执行接口、硬件部署代码或独立数据集。复现可以核对算法步骤与作者案例，但不能据此认为已复现真机规划。

## 许可说明

截至核验日，仓库根目录列表包含 README、Python 源码、Notebook 和环境配置，没有 LICENSE 文件。GitHub 上能访问源码不自动等于获得特定商用或再分发许可；有此类需求时应先联系维护者。

- **作者维护的项目页：** [HOP](../sites/hop.md)
