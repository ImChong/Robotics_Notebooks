# 小鸭子在 MotrixSim 里练起了蹬西瓜

> 来源归档

- **标题：** 小鸭子在 MotrixSim 里练起了蹬西瓜🍉
- **类型：** blog / demo（Motphys 社区传播标题；入库由维护任务触发）
- **组织：** Motphys
- **代码：** https://github.com/Motphys/MotrixLab
- **任务：** `microduck-ball-balance`（注册环境名）
- **训练命令：** `python scripts/train.py task=microduck-ball-balance/motrix.fastsac play=true`
- **入库日期：** 2026-09-13
- **一句话说明：** MotrixLab 为 Pollen Microduck 新增「双脚蹬篮球保持平衡」趣味 RL 任务；口语「蹬西瓜」指站在滚动球体上全身协调平衡，官方资产为半径 0.14 m 篮球。
- **开源状态：** **已开源** — 训练代码已合入 MotrixLab 主仓；步骤 2.5 核查 GitHub README、中文环境文档 `docs/source/zh_CN/user_guide/envs/ball_balance.md` 与 `motrix_envs/locomotion/ball_balance/` 源码均可访问。

---

## 为什么值得保留

- **低门槛体验 Motrix 栈：** 单命令 5–10 分钟即可在 MotrixSim 上看到 Microduck 学平衡，适合作为 MotrixLab / FastSAC 入门 demo。
- **与 Pollen 官方栈对照：** [microduck_rl](../repos/microduck_rl.md) 主攻行走/踢球/起身；MotrixLab 侧补齐 **球上平衡** 这一接触–平衡耦合任务，且用 **off-policy FastSAC** 而非 PPO。
- **奖励与观测可教学：** 指数核 `ball_under_feet`、特权 critic 球状态、无物理 DR 的干净设定，适合对照 [reward-design](../../wiki/concepts/reward-design.md)。

## 核心信息

| 项 | 内容 |
|----|------|
| 机器人 | Pollen Microduck，14 驱动 DoF |
| 道具 | 自由篮球（非西瓜模型；传播用语） |
| 仿真 | MotrixSim CPU/GPU 批量后端 |
| 算法 | Motrix FastSAC（`motrix.fastsac`）；该任务目前仅提供此配方 |
| 并行 | 2048 env；`play=true` 训练时实时渲染 |
| 文档 | [球平衡（中文）](https://motrixlab.readthedocs.io/zh-cn/stable/user_guide/envs/ball_balance.html) |

## 对 wiki 的映射

- `wiki/tasks/microduck-ball-balance.md` — 任务页（命令、观测奖励、与 mjlab 对照）
- `wiki/entities/motrix.md` — 平台实体补充 FastSAC 与 Microduck 环境族
- `wiki/entities/pollen-microduck.md` — 交叉链接 MotrixLab 侧训练入口
