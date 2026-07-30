# pi-r2-flow/pi-r2-flow

> 来源归档（ingest 配套仓库）

- **URL：** <https://github.com/pi-r2-flow/pi-r2-flow>
- **对应论文：** [arXiv:2607.26055](https://arxiv.org/abs/2607.26055)
- **项目页：** <https://pi-r2-flow.github.io/>
- **入库日期：** 2026-07-30
- **一句话说明：** πR² 官方实现：`deployment/` 真机栈 + `learning/Isaac-GR00T`（NVIDIA Isaac-GR00T fork，`pir2` 分支）训练三变体。
- **代码：** <https://github.com/pi-r2-flow/pi-r2-flow>（需 `--recursive` 拉 submodule）

## 仓库结构（README 快照）

| 路径 | 作用 |
|------|------|
| `deployment/apps/run_policy.py` | 主控制循环；`--query-mode {sync,pipelined,continuous}` |
| `deployment/apps/run_camera_server.py` | RealSense ZMQ 相机服务 |
| `deployment/mindex/robots/` | xArm6 / XHand 驱动 |
| `deployment/mindex/policy/groot_client.py` | GR00T ZMQ 客户端 |
| `learning/Isaac-GR00T/` | GR00T-N1.7 微调（submodule） |
| `gr00t/experiment/launch_finetune.py` | 统一训练入口；变体靠旗标区分 |

## 三变体与查询模式

| 变体 | ckpt-type | 典型 query-mode |
|------|-----------|-----------------|
| PI-R2（Ours） | `pir2` | `continuous` 或 `pipelined` + `--async-vlm` |
| Train-time RTC | `rtc` | `pipelined` + `--inpaint` |
| Standard flow | `plain_flow` | `sync` 或 continuous + temporal ensembling |

**硬件：** UFactory xArm6 + XHand（12-DoF）+ RealSense；动作 18 维。

**PI-R2 训练旗标（摘要）：** `--streaming --streaming-schedule-mode pir2 --image-delay-max 5` 等；环境变量 `GR00T_IMAGE_DELAY_MAX=5`。

## 开源边界

- **已发布：** 训练脚本、部署栈、与基线对照的运行入口。
- **自备：** `GR00T-N1.7-3B` 基座权重、LeRobot 格式微调数据、真机驱动环境。
- README 未声明统一 SPDX 许可证字段（API `license: null`）；以仓库 `LICENSE`/NOTICE 为准复核查。

## 对 wiki 的映射

- [πR² 论文实体](../../wiki/entities/paper-pi-r2.md)
- [论文归档](../papers/pi_r2_arxiv_2607_26055.md)
- [项目页归档](../sites/pi-r2-flow-github-io.md)
