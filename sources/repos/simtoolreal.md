# SimToolReal 官方代码仓库（tylerlum/simtoolreal）

> 来源归档（官方实现仓 + Humanoid Motion Intelligence 开源项目主表）

- **标题：** SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation
- **类型：** repo
- **技术路线分组：** LocoManip
- **链接：** <https://github.com/tylerlum/simtoolreal>
- **项目页：** <https://simtoolreal.github.io/>
- **论文：** <https://arxiv.org/abs/2602.16863>（RSS 2026）
- **许可证：** MIT
- **入库日期：** 2026-07-30
- **再次核对：** 2026-11-21
- **核对范围：** 通读 `tylerlum/simtoolreal`（commit `313d5ae`）官方 `README.md`、`docs/deployment.md`、`docs/dextoolbench.md`，并核对训练入口 `isaacsimenvs/train.py` 与部署节点 `deployment/rl_policy_node.py`、`deployment/goal_pose_node.py`、`deployment/sharpa_node.py` 目录结构；未下载数据集或在本地运行 Isaac Sim / Isaac Gym / 真机。
- **通读：** 已完成
- **一句话说明：** SimToolReal 官方开源仓库，提供 Isaac Sim（推荐）与 Isaac Gym（论文原始训练环境）下的 SAPG 训练、DexToolBench 24 任务评测与可视化工具，以及基于 ROS 的 Sim2Sim / Sim2Real 部署节点。
- **沉淀到 wiki：** 是 → [`wiki/entities/simtoolreal.md`](../../wiki/entities/simtoolreal.md)

---

## 仓库目录与核心入口（commit `313d5ae`）

| 目录 / 文件 | 职责与可运行入口 |
|-------------|------------------|
| `isaacsimenvs/train.py` | 推荐的 Isaac Lab / Isaac Sim 训练入口（`--task Isaacsimenvs-SimToolReal-Direct-v0 --agent rl_games_sapg_cfg_entry_point`） |
| `isaacgymenvs/launch_training.py` | 论文使用的遗留 Isaac Gym SAPG 训练入口；支持 `scripts/replay_checkpoint_gym.sh` 本地 Three.js 回放 |
| `rl_games/` | 内置修改版 PPO 与 SAPG 算法实现 |
| `dextoolbench/` | DexToolBench 评测与可视化：`run_all_evals_isaacsim.py`、`eval_interactive_isaacsim.py`、`generate_collision_meshes.py`（CoACD 凸分解） |
| `deployment/` | 4 节点 ROS 部署链路：`rl_policy_node.py`、`goal_pose_node.py`、`sharpa_node.py`、`visualization_node.py` 及 `isaac/`、`fake/` Sim2Sim 测试节点 |
| `download_pretrained_policy.py` / `download_dextoolbench_data.py` | 下载预训练策略权重（`config.yaml`、`model.pth`）及 DexToolBench RGB-D 与位姿轨迹数据 |

## 部署与外部依赖边界（`docs/deployment.md`）

- **4 节点 ROS 拓扑：** (1) `rl_policy_node.py`（订阅 `/robot_frame/current_object_pose`、`/robot_frame/goal_object_pose`、`/iiwa/joint_states`、`/sharpa/joint_states`，发布 `/iiwa/joint_cmd`、`/sharpa/joint_cmd`）；(2) `goal_pose_node.py`（依据关键点距离阈值 `dist < threshold` 步进更新目标位姿）；(3) 感知节点（依赖外部 fork [kushal2000/FoundationPose](https://github.com/kushal2000/FoundationPose) 运行 SAM + FoundationPose）；(4) 机器人节点（`iiwa_control` ROS 包与 `deployment/sharpa_node.py`）。
- **Sim2Sim 路径：** 通过 `deployment/isaac/isaac_env_node.py` 或无物理测试节点 `deployment/fake/fake_robot_node.py` + `fake_perception_node.py` 替代真机与相机节点验证闭环接口。

## 对 wiki 的映射

- [SimToolReal](../../wiki/entities/simtoolreal.md)
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
