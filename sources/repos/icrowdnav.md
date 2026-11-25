# iCrowdNav（BRoln7/icrowdnav）

> 来源归档（repo）

- **标题：** iCrowdNav — Learning Robot Visual Navigation in Crowds via Intention-Aware Scene Representations
- **类型：** repo / visual-crowd-navigation / drl / sim2real
- **来源：** BRoln7（GitHub）
- **链接：** <https://github.com/BRoln7/icrowdnav>
- **项目页：** <https://broln7.github.io/socialbev.io/>
- **论文：** [arXiv:2606.26047](https://arxiv.org/abs/2606.26047) · RA-L 2026 (`10.1109/LRA.2026.3677748`)
- **演示视频：** <https://www.youtube.com/watch?v=8q0dhAiWCEA>
- **Stars：** 见 GitHub 实时状态（`pushed_at: 2026-09-17`，322 files）
- **入库日期：** 2026-07-28
- **复核日期：** 2026-11-21
- **一句话说明：** iCrowdNav 官方开源仓（MIT，322 files）：提供基于 ROS 1 + `stable-baselines3`（`PPO`）与 Isaac Sim 4.0 / Pegasus 的训练与推理脚本（`policy_train.py`、`policy_inference.py`、`socbev_gym`、`bev_perception`、`scene/Astar`）；原始 Nucleus 商业 USD 场景与预训练权重未随仓分发。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-icrowdnav.md`](../../wiki/entities/paper-icrowdnav.md)

---

## 开源状态（步骤 2.5，2026-11-21 复核）

| 项 | 状态 |
|----|------|
| 训练 / 推理代码 | **已开源**（`icrowdnav/src/drl_policy/policy_train.py`、`policy_inference.py`、`socbev_gym/envs/soc_bev_env.py`、`bev_perception/`、`scene/Astar/`） |
| 权重 / checkpoint | **未随仓分发**（README 说明 `trained weights are not shipped in this repository yet`） |
| 仿真场景资产 | 部分开放 — 含仓库场景配置与 Pegasus Simulator（BSD-3-Clause）；论文用 Nucleus USD 商业场景受分发协议限制未包含 |
| 许可证 | **MIT License**（第三方如 Pegasus 为 BSD-3-Clause、`pedsim_msgs` 为 BSD） |

**结论：** **源码已开源（MIT）**，训练/推理与 `social-bev-v0` Gym 环境均可读可运行；复现论文全部仿真世界需自备 Isaac Sim 场景与重训 PPO 权重。wiki 已补基于实际源码的「源码运行时序图」。

---

## 技术栈与入口核对

| 组件 | 实仓实现 |
|------|----------|
| 仿真 | Isaac Sim 4.0 + Pegasus Simulator + ROS Noetic（`icrowdnav/src/scene/`） |
| 训练 / 环境 | `gym.make("social-bev-v0")` → `SocBevEnv`；`stable_baselines3.PPO("MultiInputPolicy", ...)`（`policy_train.py`） |
| 感知特征 | `bev_perception/bev_perception.py`、`models/{bev_generator,humanpose_attention,temporal_model}.py` |
| 评估与部署 | `rosrun drl_policy policy_inference.py`（读取 `--checkpoint` 与 `warehouse.yaml`） |

---

## 目录快照（2026-11-21 复核，322 files）

```
icrowdnav/
  README.md / LICENSE
  icrowdnav/src/
    drl_policy/
      policy_train.py
      policy_inference.py
      config/default.yaml
      socbev_gym/envs/soc_bev_env.py
      bev_perception/{bev_perception.py, models/, layers/}
      mdp/{reward.py, utils.py}
    scene/{Astar/, PegasusSimulator/}
```

---

## 交叉链接

- 论文归档：[`sources/papers/icrowdnav_arxiv_2606_26047.md`](../papers/icrowdnav_arxiv_2606_26047.md)
- 项目页：[`sources/sites/broln7-socialbev-io.md`](../sites/broln7-socialbev-io.md)
- wiki 实体：[`wiki/entities/paper-icrowdnav.md`](../../wiki/entities/paper-icrowdnav.md)
