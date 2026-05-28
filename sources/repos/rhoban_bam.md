# BAM（Better Actuator Models）

> 来源归档（仓库 README 与公开文档要点摘录，非全文镜像）

- **标题：** BAM: Better Actuator Models
- **类型：** repo
- **组织：** Rhoban（GitHub）
- **链接：** <https://github.com/Rhoban/bam>
- **许可：** Apache-2.0
- **论文：** [arXiv:2410.08650](https://arxiv.org/abs/2410.08650v1)（ICRA 2025）
- **入库日期：** 2026-05-28
- **一句话说明：** 开源摆锤台架数据采集、后处理、CMA-ES 摩擦/电机参数拟合与 MuJoCo 2R 臂验证管线；内置 M1–M6 模型与 Dynamixel / eRob 示例。
- **沉淀到 wiki：** [wiki/entities/bam-better-actuator-models.md](../../wiki/entities/bam-better-actuator-models.md)、[wiki/entities/paper-bam-extended-friction-servo-actuators.md](../../wiki/entities/paper-bam-extended-friction-servo-actuators.md)

---

## 仓库结构（功能面）

| 模块 | 路径 / 命令 | 作用 |
|------|-------------|------|
| 摆锤采集（Dynamixel） | `python -m bam.dynamixel.record` | 串口记录 sin_time_square / sin_sin / lift_and_drop / up_and_down 等轨迹 |
| 摆锤采集（eRob + Etherban） | `python -m bam.erob.record` | gRPC 经 Etherban 服务器；需 `generate_protobuf.sh` |
| 后处理 | `python -m bam.process --dt 0.005` | 线性插值到固定步长 |
| 参数拟合 | `python -m bam.fit --model m6 --method cmaes` | 输出 `params/<actuator>/m*.json` |
| 对比绘图 | `python -m bam.plot --sim --params ...` | 实测 vs 仿真曲线 |
| Drive/Backdrive 图 | `python -m bam.drive_backdrive` | 可视化摩擦上界与负载相关形状 |
| 2R 验证 | `2R/` + `python -m 2R.sim` | MuJoCo URDF、record_2R、mae.sh |

**已发布辨识数据：** [Google Drive](https://drive.google.com/drive/folders/1SwVCcpJko7ZBsmSTuu3G_ZipVQFGZ11N?usp=drive_link)（MX-64 / MX-106 / eRob80:50 / eRob80:100）。

---

## 依赖

- 辨识：`requirements_bam.txt`（含 optuna / CMA-ES 等）
- 2R 仿真：`requirements_2R.txt`（MuJoCo）

---

## 与论文的对应关系

- README 中 **M1–M6** 命名与论文一致；默认示例对 MX-106 拟合 **m6**，2R Dynamixel 验证推荐 **m4** 参数对。
- 视频教程：<https://youtu.be/5XPEEKDnQEM>

---

## 对 wiki 的映射

- [BAM 仓库实体](../../wiki/entities/bam-better-actuator-models.md)
- [扩展摩擦论文实体](../../wiki/entities/paper-bam-extended-friction-servo-actuators.md)
- [Actuator Network](../../wiki/methods/actuator-network.md) — 数据驱动执行器建模对照
- [SAGE](../../wiki/entities/sage-sim2real-actuator-gap-estimator.md) — 另一套「仿真–真机执行器 gap 度量」工具链
