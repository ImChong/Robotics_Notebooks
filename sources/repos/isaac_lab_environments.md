# isaac_lab_environments

> 来源归档

- **标题：** Isaac Lab — Available Environments（官方默认任务清单）
- **类型：** repo + 官方文档
- **来源：** NVIDIA（isaac-sim 组织）
- **链接：** https://github.com/isaac-sim/IsaacLab
- **文档：** https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html
- **一手文档源码：** `docs/source/overview/environments.rst`
- **任务源码：** `source/isaaclab_tasks/isaaclab_tasks/{manager_based,direct}/`
- **入库日期：** 2026-08-17
- **核对版本：** `VERSION` = **3.0.0**；`main` 分支 commit `2e44ddb`（2026-08-10）
- **一句话说明：** Isaac Lab 随框架自带的全部 RL / IL 任务注册表，是「开箱即跑」的默认环境集合，也是 locomotion / manipulation 基线复现的起点。
- **代码：** https://github.com/isaac-sim/IsaacLab（已开源，BSD-3-Clause）
- **沉淀到 wiki：** 是 → [`wiki/entities/isaac-lab-default-environments.md`](../../wiki/entities/isaac-lab-default-environments.md)

---

## 为什么值得保留

- Isaac Lab 的默认环境是**事实上的 GPU 并行 RL 基线集**：ANYmal / Go2 / G1 / H1 / Spot / Digit 的 velocity 任务、Franka / UR10 的操作任务、Factory-FORGE-AutoMate 的接触密集装配，大量论文直接在这些 ID 上做对照。
- 官方文档页只列「代表性任务」，**真正的全量清单在 `gym.register` 里**；两者数量差一个量级，选型时容易漏看（例如 `Deploy-*`、`Assemble-Trocar-*`、`Lift-Cloth/Soft`、Spaces Showcase 系列）。
- 任务 ID 的命名法本身编码了工作流（Manager-Based / Direct）、控制空间（joint pos / IK-Abs / IK-Rel / OSC / RmpFlow / Pink-IK）、地形（Flat / Rough）与用途（Play / ROS-Inference / Mimic），是读懂 Isaac Lab 生态的索引。

---

## 一手核对方法（可复现）

```bash
# 1) 稀疏 clone 官方仓（只取任务与文档，约 13 MB）
git clone --depth 1 --filter=blob:none --sparse https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git sparse-checkout set source/isaaclab_tasks scripts docs/source/overview

# 2) 扫描全部 gym.register 注册项（本次核对用的方式，无需装 Isaac Sim）
#    正则匹配 gym.register(...) 块中的 id= 与 *_cfg_entry_point

# 3) 装好 Isaac Sim 后，官方等价命令：
./isaaclab.sh -p scripts/environments/list_envs.py                  # 全量清单
./isaaclab.sh -p scripts/environments/list_envs.py --keyword G1     # 关键字过滤
./isaaclab.sh -p scripts/environments/list_envs.py --show_presets   # 附带 preset 选择器
```

---

## 核对结论（commit `2e44ddb`）

| 指标 | 数值 |
|------|------|
| 注册任务 ID 总数 | **197**（另有 `My-Awesome-Task-v0` 为 `utils/parse_cfg.py` 文档示例，不计入） |
| 基础任务 / Play 推理变体 | **153** / **44** |
| Manager-Based / Direct | **139** / **58** |
| 带 rsl_rl 配置 | 107 |
| 带 skrl 配置 | 104（含 AMP 3、IPPO 2、MAPPO 2） |
| 带 rl_games 配置 | 66 |
| 带 sb3 配置 | 10 |
| 带 robomimic BC（IL）配置 | 17 |
| 无内置 RL 配置（遥操作 / 数据生成 / 控制空间变体） | 24 |

- **文档页 vs 源码的差异：** 网页版 `environments.html` 未覆盖 `Isaac-Deploy-GearAssembly-*`、`Isaac-Assemble-Trocar-G129-Dex3-*`、`Isaac-NutPour/ExhaustPipe-GR1T2-Pink-IK-Abs-*`、`Isaac-Lift-Cloth/Soft-Franka-*`、`Isaac-Cartpole-{Albedo,SimpleShading-*}-Camera-Direct-*` 等条目；仓内 `.rst` 的自动生成表（`START-AUTO-GENERATED: comprehensive-environment-list`）比网页更新更快。
- **3.0 新增：** `Preset Selectors` 机制（`physics=` / `renderer=` / `presets=`），locomotion 与 classic 任务已可切 `newton_mjwarp`、`newton_kamino`、`ovphysx` 物理后端；Digit 因闭链结构仍须 `physx`。
- **训练脚本目录：** `scripts/reinforcement_learning/{rsl_rl,rl_games,skrl,sb3,ray,rlinf,leapp}`、`scripts/imitation_learning/{isaaclab_mimic,robomimic,locomanipulation_sdg}`、`scripts/sim2sim_transfer/`。

---

## 关联档案

- 框架主档：[`isaac_lab.md`](./isaac_lab.md)
- 仿真底座：[`isaac_sim.md`](./isaac_sim.md)
- 教学任务：[`sources/sites/isaac-lab-cartpole.md`](../sites/isaac-lab-cartpole.md)
