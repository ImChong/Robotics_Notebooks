# letools_opensource（LeTools Skills / 上位机技能工具链）

> 来源归档（repo）

- **标题：** LejuRobotics/letools_opensource
- **类型：** repo
- **代码：** <https://github.com/LejuRobotics/letools_opensource>
- **镜像/Issues 叙事：** README 引导外部用户走 [GitCode OpenLET Issues](https://gitcode.com/OpenLET/letools_opensource/issues)
- **文档：** <https://www.letools.lejurobot.com/docs.html?type=skills>
- **语言：** 以 C++/Python/ROS 混合为主（GitHub 主语言标注 C++）
- **Stars（入库快照）：** 3
- **创建：** 2026-07-23 · **默认分支：** `main`
- **许可：** GitHub API **无 SPDX**；README 写「由乐聚机器人维护」，**未**给出 GPL/Apache/MIT 文本
- **入库日期：** 2026-08-17
- **一句话说明：** Kuavo **上位机侧**分层框架：`core` 接口 → `adapters` 硬件 → **`skills` 原子技能** → `orchestration` 行为树 → `apps` 示例；支持 dry-run / MuJoCo / 真机。

## 开源状态（步骤 2.5）

- **已开源、可运行：** 公开仓含安装脚本、行为树 dry-run、SDK 示例与 `skills/atomic/refactored_sdk/` 主力技能。
- **边界：**
  - 依赖 `kuavo_humanoid_sdk` 子模块（`scripts/install_sdk.sh` 锁定 `kuavo-ros-opensource` 分支/tag，如 README 示例 `master` / `1.4.4`）；子模块源为 gitcode，网络失败是高频坑。
  - `adapters/hardware/leju_bipedal/` 标明 **后续扩展**，当前主力是 **`leju_wheeled`**。
  - 旧技能目录 `manipulation/` `motion/` `grasp_skill.py` 仍服务 `smoke_v1`，新场景应走 `refactored_sdk/`。
  - **许可证未在 GitHub 元数据声明**，商用分发前需向乐聚确认。

## 分层（README 归纳）

```text
apps 示例
  → orchestration 行为树（Node = py_trees 生命周期 + JSON 参数）
    → skills 原子技能（业务动作；不直接发 ROS topic）
      → adapters/hardware（标准接口 / *_sdk / TimedCmd）
        → kuavo_humanoid_sdk / ROS
```

**Skill vs Node：** Node 接行为树；Skill 接 `hardware.xxx()`。禁止在 Skill 里直接 `Publisher`/`ServiceProxy`。

### 新架构技能（`skills/atomic/refactored_sdk/`）

`SkillBase` 生命周期：`on_initialize` → `on_execute`（可能 50 Hz 多次）→ `on_is_finished`。

主力文件包括：底盘本体系位姿、头部 SDK、双臂 14 关节轨迹、末端局部/世界系轨迹、手臂复位、腿部关节、等待秒/Enter 等。

### 三种硬件控制路径

| 路径 | 特征 | 场景 |
|------|------|------|
| 标准接口 | `send_base_pose` / `control_head` 等 | Skill、行为树 |
| SDK 直调 | `*_sdk` | 高频、底层验证 |
| TimedCmd | `*_timed`；Ruckig / IK / 离线轨迹 | 带时间规划 |

## 环境要点

- Ubuntu 20.04 + ROS Noetic + Python 3.8+ + catkin
- 推荐先 `catkin build` `infrastructure/ros_packages`，再 `scripts/install_sdk.sh`
- 无硬件时：`run_behavior_tree_json.py --scenario …/refactored_sdk_atomic_v1 --dry-run --tick-once`

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 平台枢纽 | [wiki/entities/letools.md](../../wiki/entities/letools.md) |
| 行为树对照 | [wiki/concepts/behavior-tree-vla-orchestration.md](../../wiki/concepts/behavior-tree-vla-orchestration.md) |
| 硬件运营方 | [wiki/entities/leju-robotics.md](../../wiki/entities/leju-robotics.md) |
| 训练栈（勿混） | [letools-learning.md](letools-learning.md) |

## 关键源文件

- 仓 README：<https://github.com/LejuRobotics/letools_opensource/blob/main/README.md>
- 原子技能层：<https://github.com/LejuRobotics/letools_opensource/blob/main/skills/README.md>
