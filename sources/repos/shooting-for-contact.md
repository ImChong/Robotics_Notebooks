# sesteban951/shooting-for-contact

> 来源归档（ingest · 2026-08-07）

- **标题：** Shooting for Contact — Direct Simulation-based Multiple Shooting（DSMS）参考实现
- **类型：** repo
- **组织 / 作者：** [sesteban951](https://github.com/sesteban951)（Sergio A. Esteban 等）
- **URL：** <https://github.com/sesteban951/shooting-for-contact>
- **Homepage：** <https://shooting-for-contact.github.io/>
- **License：** 截至入库日 GitHub **未列出** SPDX LICENSE 文件（以仓库为准）
- **Stars：** ~12（2026-08-07，以 GitHub 为准）
- **Language：** Python
- **入库日期：** 2026-08-07
- **一句话说明：** 论文 **Shooting for Contact**（arXiv:2608.03116）的 **DSMS** 实现：MuJoCo 在环多重打靶 NLP（IPOPT/`cyipopt`）+ receding-horizon MPC；含 Unitree G1 / Go2 与玩具系统示例。
- **沉淀到 wiki：** [`wiki/entities/paper-shooting-for-contact.md`](../../wiki/entities/paper-shooting-for-contact.md)、[`wiki/methods/dsms-contact-implicit-multiple-shooting.md`](../../wiki/methods/dsms-contact-implicit-multiple-shooting.md)

---

## 核心定位

本仓发布 **接触隐式轨迹优化 / 重定向** 核心求解器与可复现示例，**不是**完整 mjlab RL 训练或真机部署栈。动力学、接触与导数来自 MuJoCo；NLP 在 `src/multi_shooting.py`，MPC 包装在 `src/mpc.py`。

```
src/
  multi_shooting.py   # NLP：决策向量、defect、IPOPT callbacks
  mpc.py              # receding-horizon 包装
  dynamics.py         # MuJoCo rollout / FD Jacobian / manifold 状态
  spline.py           # ZOH / 线性控制参数化
  end_effector.py     # 笛卡尔 body tracking
utils/                # quaternion-aware 状态、.npz I/O
models/               # unitree_g1、unitree_go2、cartpole、hopper 等 XML
trajectories/         # g1/、go2/ 参考 clips + 生成脚本
examples/             # 每题一目录；求解写 .npz，可用 replay.py 回放
```

---

## 依赖栈（environment.yml）

| 包 | 版本（仓内锁定） |
|----|------------------|
| Python | 3.11 |
| numpy | ≥2.0 |
| scipy | ≥1.15 |
| cyipopt | ≥1.5 |
| mujoco | ≥3.8 |
| matplotlib | ≥3.8 |

安装：`make install` → conda env `dsms`，并设置 `TRAJOPT_ROOT_DIR`。

---

## 用法要点（README）

### 玩具系统

```bash
conda activate dsms
python examples/cartpole/cartpole.py
python examples/replay.py cartpole
python examples/cartpole_mpc/cartpole_mpc.py
python examples/hopper_mpc/hopper_mpc.py
```

### Unitree Go2

```bash
python examples/go2_tracking_mpc/go2_tracking_mpc.py hopturn
python examples/replay.py go2_tracking_hopturn --speed 0.5
```

### Unitree G1

```bash
python examples/g1_squat_mpc/g1_squat_mpc.py
python examples/g1_gait/g1_gait.py run_fwd   # walk_fwd | run_fwd | run_bck | crawl_fwd
python examples/g1_tracking_mpc/g1_tracking_mpc.py crawl_fwd
python examples/replay.py g1_tracking_crawl_fwd --speed 0.5
```

- `actuator_mode`：`"torque"` 或 `"position"`（PD 目标）。
- 新运动：在 `trajectories/g1/...` 放 `qpos_*dof.csv` + `time.csv`，并在 example `config.py` 注册 `traj_dir`。

---

## 与论文管线的边界

| 模块 | 本仓 | 论文 / 项目页 |
|------|------|----------------|
| DSMS trajopt / MPC | ✅ | ✅ |
| Gait library 合成示例 | ✅（`g1_gait` 等） | ✅ |
| mjlab PPO / asymmetric AC | ❌ | ✅ |
| 真机部署 | ❌ | ✅（项目页视频） |

## 对 wiki 的映射

- [`wiki/entities/paper-shooting-for-contact.md`](../../wiki/entities/paper-shooting-for-contact.md)
- [`wiki/methods/dsms-contact-implicit-multiple-shooting.md`](../../wiki/methods/dsms-contact-implicit-multiple-shooting.md)
- [`sources/papers/shooting_for_contact_arxiv_2608_03116.md`](../papers/shooting_for_contact_arxiv_2608_03116.md)
- [`sources/sites/shooting-for-contact-github-io.md`](../sites/shooting-for-contact-github-io.md)
