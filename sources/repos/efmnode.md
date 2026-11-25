# EFMNode

> 来源归档（国内具身开源全景）

- **标题：** EFMNode
- **类型：** repo
- **机构：** 星海图
- **链接：** https://github.com/OpenGalaxea/EFMNode
- **分类：** 部署运行时
- **入库日期：** 2026-09-06
- **核对范围：** 已克隆 `OpenGalaxea/EFMNode@154b69c`（核对 `README.md`、`run.py`；未连接真实 R1_LITE/R1_PRO 机器人与 ROS2 Humble/TensorRT 硬件执行）。
- **通读：** 已完成
- **一句话说明：** 星海图（Galaxea AI）开源的具身基础模型真机部署运行时（ROS2 Humble 节点 + `glx` CLI / Docker），围绕 `GalaxeaFM` 模型权重提供 PyTorch 与 TensorRT 推理及 `R1_LITE`/`R1_PRO` 控制调度。
- **沉淀到 wiki：** [`wiki/entities/cn-os-efmnode.md`](../../wiki/entities/cn-os-efmnode.md)

## 开源状态

- **已开源**：部署节点代码已公开（<https://github.com/OpenGalaxea/EFMNode>，仓库根目录未单列 LICENSE 文件；需配合 `GalaxeaFM` checkpoint 权重与 `efmnode.toml` 使用）。

## 一手仓库要点摘录（`OpenGalaxea/EFMNode@154b69c`）

- **启动与调度链路**：支持 `glx` CLI（Docker 镜像）或本地运行 `scripts/run.sh` → `run.py` → `Scheduler(config).run()`，按 `control_frequency`（默认 15 Hz）与 `step_mode`（`sync` / `async`）执行推理与真机发布。
- **推理后端与录制**：读取模型目录下的 `config.yaml`、`dataset_stats.json` 与 `efmnode.toml`，支持 PyTorch 原生或 TensorRT 加速（依赖自定义算子库 `gemma_rmsnorm.so`）；提供 `scripts/record.sh` 录制 ROS2 话题。

## 对 wiki 的映射

- [wiki/entities/cn-os-efmnode.md](../../wiki/entities/cn-os-efmnode.md)
