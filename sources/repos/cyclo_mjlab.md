# cyclo_mjlab

> 来源归档

- **标题：** cyclo_mjlab — MuJoCo / MJLab RL for ROBOTIS K1
- **类型：** repo
- **组织：** [ROBOTIS-GIT](https://github.com/ROBOTIS-GIT)
- **链接：** <https://github.com/ROBOTIS-GIT/cyclo_mjlab>
- **许可：** Apache-2.0
- **Stars：** ~3（2026-09-05，新仓）
- **入库日期：** 2026-09-05
- **一句话说明：** 基于 [mjlab](https://github.com/mujocolab/mjlab) + MuJoCo 3.5.0 的 ROBOTIS K1 人形 RL/模仿学习环境：平地速度跟踪（Velocity）与参考动作模仿（Mimic / Dance1–2）；训练自动写 `params/sim2real.yaml`，play 导出 `exported/policy.onnx` / `.pt`；Docker 一键环境（MJLab 1.2.0 · Python 3.11）。

## 入口速查（README · 2026-09-05）

| 路径 / 能力 | 作用 |
|-------------|------|
| Docker 安装 | 预装 MuJoCo / MJLab / GPU 配置，无需主机 Conda |
| **Velocity** | K1 平地速度跟踪 locomotion |
| **Mimic** | Dance1 / Dance2 参考动作跟踪 |
| `params/sim2real.yaml` | 训练启动时自动生成 Sim2Real 参数 |
| `exported/policy.onnx` / `.pt` | play 阶段自动导出部署权重 |

## 与 Cyclo 栈关系

- **仿真路径对照：** [cyclo_lab](cyclo_lab.md) 走 **Isaac Lab**；本仓走 **MuJoCo Warp / mjlab**，面向 **AI Sapiens K1** 而非 OMY/FFW 臂系任务。
- **下游真机：** 导出 ONNX 后需对接 K1 bringup（`ai_sapiens` / `cyclo_control`）；与 Isaac 路径的 DDS Sim2Real 脚本不同，须按本仓 `sim2real.yaml` 与部署文档对齐。

## 开源状态

**已开源**（Apache-2.0，README 含 Docker 与任务说明；仓较新，生态仍在扩展）。

## 对 wiki 的映射

- [robotis-cyclo-mjlab.md](../../wiki/entities/robotis-cyclo-mjlab.md)
- 组织 hub：[robotis.md](../../wiki/entities/robotis.md)
- 框架对照：[mjlab.md](../../wiki/entities/mjlab.md)、[cyclo-lab.md](../../wiki/entities/cyclo-lab.md)
