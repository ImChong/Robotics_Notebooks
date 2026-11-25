# Diffusion-Planner 官方开源实现归档

- **类型：** repo / pytorch / nuplan / diffusion / planner
- **官方仓库：** <https://github.com/ZhengYinan-AIR/Diffusion-Planner>
- **原文：** <https://arxiv.org/html/2501.15564v2>
- **项目：** <https://zhengyinan-air.github.io/Diffusion-Planner/>
- **模型权重：** <https://huggingface.co/ZhengYinan2001/Diffusion-Planner>（README 指定 `args.json`、`model.pth`）
- **入库日期：** 2026-11-21
- **核对范围：** 阅读官方 README 全文与 `data_process.sh`、`torch_run.sh`、`sim_diffusion_planner_runner.sh`、`diffusion_planner/planner/planner.py`、`diffusion_planner/model/diffusion_planner.py`、`diffusion_planner/model/module/decoder.py`、`diffusion_planner/train_epoch.py`、`diffusion_planner/model/guidance/documentation_guidance.md` 的实际入口和数据流；不是声称逐行审计整个代码仓。
- **通读：** 已完成
- **开源状态：** 已开源训练/数据处理/nuPlan 仿真评测/引导 demo 及公开 checkpoint；数据需要自行获取 nuPlan，配送车 200 h 私有数据仍待发布。

## 官方入口与调用链（静态核对）

| 入口/模块 | 仓库实际行为 |
|---|---|
| `data_process.sh` | 调用 `python data_process.py --data_path --map_path --save_path --total_scenarios 1000000` |
| `torch_run.sh` | 8 进程 `torch.distributed.run ... train_predictor.py`；填写训练数据路径 |
| `sim_diffusion_planner_runner.sh` | 配 nuPlan 数据/地图/exp、`SPLIT` 与 `CHALLENGE`，通过 nuPlan 的 `run_simulation.py` 与 Hydra `planner=diffusion_planner` 启动闭环 |
| `DiffusionPlanner.initialize` | 从 `model.pth` 读取模型（默认 EMA 权重）并设为 eval |
| `DiffusionPlanner.compute_planner_trajectory` | `DataProcessor.observation_adapter` → 观测归一化 → `Diffusion_Planner` → `InterpolatedTrajectory` |
| `Diffusion_Planner.forward` | 调用 `Encoder` 和 `Decoder` |
| `Decoder.forward`（推理） | 构造当前 ego/邻车状态与噪声未来，`dpm_sampler` 联合生成；可选 `_guidance_fn`，最后逆归一化 |
| `outputs_to_trajectory` | ego 轨迹从 (x,y,cos,sin) 转为 (x,y,heading)，调用 nuPlan `transform_predictions_to_states` |

## 最短复现与限制

1. 需先依照官方 README 配 Python 3.9、nuplan-devkit、仓库依赖，并安装 nuPlan 数据与地图；路径需按本机调整。
2. 下载 `checkpoints/args.json` 与 `checkpoints/model.pth`；配置 `sim_diffusion_planner_runner.sh` 中 4 个环境变量、`SPLIT`、`CHALLENGE`。
3. 执行 `bash sim_diffusion_planner_runner.sh`，通过 `run_nuboard.ipynb` 看结果；可用 `bash sim_guidance_demo.sh` 检验引导，但须先按官方文档配置。
4. 训练则依次 `./data_process.sh`、`./torch_run.sh`，依赖原始 nuPlan 数据及多 GPU 环境；**此次仅静态核对代码，未下载数据、运行实验或复现论文数值**。

## 关联与原文

- [唯一论文详情页](../../wiki/entities/paper-diffusion-planner-autonomous-driving.md)
- [论文归档](../papers/diffusion_planner_arxiv_2501_15564.md)
- [官方项目页归档](../sites/diffusion-planner.md)
