# TacO（TacObench/TacO）

> 来源归档

- **标题：** TacO — Benchmarking Tactile Sensors for Object Manipulation
- **类型：** repo
- **组织：** UC San Diego / CMU / SNU（见论文作者）
- **代码：** <https://github.com/TacObench/TacO>
- **项目页：** <https://tacobench.github.io/>
- **论文：** <https://arxiv.org/abs/2605.21976>
- **硬件 STL：** <https://github.com/TacObench/TacObench.github.io/tree/main/3D_part_files>
- **入库日期：** 2026-07-27
- **一句话说明：** TacO 官方实现：跨模态触觉传感器上的 **ACT 模仿学习训练 / 远程部署**，以及 **硬件可重复性测试** 脚本；配套项目页仓库中的夹爪与传感器安装 3D 零件。
- **沉淀到 wiki：** [TacO 触觉传感器基准（论文实体）](../../wiki/entities/paper-taco-tactile-sensor-benchmark.md)
- **名称消歧：** 本仓 **TacO（传感器基准）** ≠ [TACO 触觉 WM](../../wiki/entities/paper-taco-tactile-wm-vla-posttrain.md)。

## 开源状态（核查，2026-07-27）

| 资产 | 状态 |
|------|------|
| 训练 / 推理代码 | **已开源**（`tactile_policy/`） |
| 传感器驱动与可重复性测试 | **已开源**（`tactile_sensors/`、`hardware_repeatability/`） |
| 夹爪 / 安装 STL | **已开源**（项目页仓 `3D_part_files/`） |
| 示范数据 / checkpoint | **截至入库日未见公开下载链**（README 仅规定 HDF5 格式） |
| License | 顶层未声明 SPDX（仓库 `license` 字段为空） |

## 仓库导航（对齐时序图节点）

| 路径 | 作用 |
|------|------|
| `create_env.sh` / `requirements.txt` | 建 conda 环境 `tactile_bench`（Python 3.11） |
| `tactile_policy/main.py` | ACT 训练入口（`--model_cfg` / `--dataset_json` / `--chunk_size`） |
| `tactile_policy/configs/sensors/*.yaml` | 传感器模态：`array` / `image` / `none`（vision-only） |
| `tactile_policy/configs/models/act_*.yaml` | 各传感器 ACT 模型配置 |
| `tactile_policy/modeling/modeling_act.py` | ACT / CVAE 策略实现 |
| `tactile_policy/remote_inference/serve_act_policy.py` | WebSocket 策略服务 |
| `tactile_policy/remote_inference/export_act_to_jit.py` | TorchScript 导出 |
| `hardware_repeatability/run_repeatability_test.py` | Dynamixel + 压头可重复性测试 |
| `tactile_sensors/` | 传感器串口 / 发布侧辅助 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Tactile Sensing](../../wiki/concepts/tactile-sensing.md) | **四模态六传感器** 真机对比，补「选型」证据 |
| [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md) | 同数据 vision-only vs visuotactile 消融 |
| [Action Chunking](../../wiki/methods/action-chunking.md) | 策略骨干为 **ACT**（chunk 64 / 执行 32） |
| [Imitation Learning](../../wiki/methods/imitation-learning.md) | GELLO / Factr 遥操作示范 → BC |
| [VTAP Gripper](../../wiki/entities/paper-vtap-gripper.md) | 同用 **FlexiTac** 指尖阵列 |

## 为何值得保留

- 把「该买哪种触觉传感器」从 anecdata 变成 **可复现真机 IL 对比**。
- 代码模块边界清晰，适合写 wiki **源码运行时序图** 与工程复现 checklist。
