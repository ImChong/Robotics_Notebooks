# yunhaif/regrind

> 来源归档（复核 2026-09-09）

- **标题：** REGRIND — Official Implementation
- **类型：** repo
- **组织：** yunhaif（Cornell / Amazon FAR 作者）
- **代码：** <https://github.com/yunhaif/regrind>
- **论文：** <https://arxiv.org/abs/2607.11874>
- **项目页：** <https://www.yunhaifeng.com/REGRIND/>
- **许可证：** MIT
- **Stars：** ~100（2026-09-09）
- **入库日期：** 2026-07-16；**复核：** 2026-09-09
- **一句话说明：** Isaac Lab 2.3.0 / Isaac Sim 5.1.0 官方实现：预计算 interaction mesh 重定向轨迹、Drake 重定向脚本、RSL-RL 残差策略训练与 Play 评测；`import regrind` 不强制 Drake。
- **沉淀到 wiki：** [`wiki/methods/regrind-retargeting-guided-rl.md`](../../wiki/methods/regrind-retargeting-guided-rl.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（MIT） |
| **推理/训练** | `scripts/rsl_rl/train.py` / `play.py`；四任务环境 + `-Play-v0` 评测环境 |
| **重定向** | `scripts/retarget_hand_object.py`；**可跳过**（仓内已有 `.h5`） |
| **重定向依赖** | 可选 `pip install -e "source/regrind[retargeting]"`（Drake）；默认 MOSEK，可 `solver=clarabel` |
| **硬件** | README 面向 LEAP / WUJI + UR5e 真机部署文档在论文；仓以仿真训练为主 |

## README 要点（2026-09-09）

### 环境

- Python **3.11**；**Isaac Sim 5.1.0** + **Isaac Lab 2.3.0**（按官方 binary 安装）
- `cd IsaacLab && ./isaaclab.sh --conda regrind`
- `cd regrind && conda activate regrind && isaaclab -i rsl_rl && pip install -e source/regrind && source scripts/set_path.sh`

### 重定向

```bash
python scripts/retarget_hand_object.py --robot {leaphand,wujihand} --object {scissors,screwdriver}
python scripts/replay_retargeted_traj.py --task Regrind-LeapHand-Scissors-Play-v0 \
  --headless --video --num_envs 1 --retargeted_traj_path /path/to/out.h5
```

输出 `.h5` 键：`robot_pos/quat/joints`、`object_pos/quat/joint`、`robot_keypoints`、`mano_joint_coords`。

### RL 训练与评测

| 脚本 | 用途 |
|------|------|
| `scripts/list_envs.py` | 列出 `Regrind-{LeapHand,WujiHand}-{Scissors,Screwdriver}-v0` |
| `scripts/rsl_rl/train.py` | PPO 训练（例：`--task Regrind-LeapHand-Scissors-v0 --headless --num_envs 4096`） |
| `scripts/rsl_rl/play.py` | 评测/录视频（用 `-Play-v0` 环境；`--auto_gravity_from_ckpt` 配合重力课程） |
| `scripts/zero_agent.py` / `random_agent.py` | 环境冒烟 |

## 对 wiki 的映射

- 方法页（兼论文索引）：[`wiki/methods/regrind-retargeting-guided-rl.md`](../../wiki/methods/regrind-retargeting-guided-rl.md)
- 论文摘录：[`sources/papers/regrind_arxiv_2607_11874.md`](../papers/regrind_arxiv_2607_11874.md)
- 项目页：[`sources/sites/regrind-project-yunhaifeng.md`](../sites/regrind-project-yunhaifeng.md)
