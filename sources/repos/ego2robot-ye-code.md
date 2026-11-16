# Ego2Robot 官方可执行代码｜www-Ye/ego2robot_blog

- **代码：** https://github.com/www-Ye/ego2robot_blog
- **项目主页：** https://www-ye.github.io/ego2robot_blog/
- **论文：** https://arxiv.org/abs/2608.02580v1
- **论文归档：** [Ego2Robot 原文核查](../papers/ego2robot_arxiv_2608_02580.md)
- **站点归档：** [官方项目主页](../sites/ego2robot-blog.md)
- **wiki：** [Ego2Robot 实体](../../wiki/entities/paper-ego2robot.md)
- **代码核对日期：** 2026-11-12（官方 `main` 分支；无固定 release 版本，后续可能变化）

## 开源状态更新

2026-08-15 的历史记录称该仓库只有静态站；**现在不再准确**。2026-11-12 核查时，官方仓库根目录包含 `ego2robot/`、`configs/`、`scripts/`、`evaluation/robotwin/`、`pyproject.toml`、README。可以静态读取合成管线和 RoboTwin 评测相关代码，不等同于全部实验数据、预训练权重或模型依赖均已打包公开。仓库根目录未见明确的顶级 LICENSE 文件，不推断整仓许可证。

## 官方代码入口与已核对调用链

| 文件 | 入口 / 实际用途 |
|---|---|
| `scripts/run_pipeline.py` | 读取 `--config` / `--mode` / `--demo-dir` / `--all-robots` 等 CLI，OmegaConf 合并 YAML / 覆盖参数，调用 `ego2robot.pipeline.run_parallel(cfg)` |
| `ego2robot/pipeline.py` | `STAGE_DEFS` 注册任务名→Processor；`resolve_modes` 从 config.mode 确定阶段；`get_demo_folders` 枚举样本 |
| `ego2robot/pipeline.py:run_parallel` | 针对每个阶段筛选已经处理的 demo、按长度分块，经 joblib loky 分发 `_process_chunk`，可按 GPU 和 worker 配额并行；失败记录在 JSONL 中 |
| `ego2robot/pipeline.py:_process_chunk` | `load_processor_class(stage_name)` 动态加载实现，实例化后调用 `proc.process_one_demo(dp)`；各阶段以存盘中间产物衔接 |
| `ego2robot/processors/data_sources/*` | 已标注 EgoDex、ViTRA、EgoVerse 数据源处理，与原始视频阶段区分 |
| `ego2robot/processors/depth_pose_processor.py`、`hand_processor.py` | 未标注原始视频路径的相机深度/姿态和手部估计相关模块 |
| `ego2robot/processors/action_processor.py`、`smoothing_processor.py` | 手→抓取动作与轨迹平滑 |
| `ego2robot/processors/segmentation_processor.py`、`handinpaint_processor.py`、`robot_inpaint/processor.py` | 人臂分割、视频修补、机器人覆盖合成和输出 |
| `scripts/convert_to_lerobot.py` | 后续独立的 NPZ/MP4→LeRobot 转换入口：检查 `valid`、对齐视频帧、读相机外参，输出 LeRobot 数据集；不可误写成 `run_parallel` 内置调用 |
| `evaluation/robotwin/` | 论文 RoboTwin 评测相关代码，独立环境；不是合成管线 `run_parallel` 的内置阶段 |

## README 公开运行方法（静态核对，未实际运行）

```bash
python scripts/run_pipeline.py --config configs/egoverse_mecka.yaml
python scripts/run_pipeline.py --config configs/egoverse_mecka.yaml --all-robots
python scripts/convert_to_lerobot.py --data_root /path/to/pipeline_output/dataset --output /path/to/lerobot_out --dataset_name ego2robot_dataset
```

`configs/egoverse_mecka.yaml` 的执行阶段为 `base_egoverse → action → smoothing → arm_segmentation → hand_inpaint → robot_inpaint`；raw-video 路径由 README 定义为 `subtask → depth_pose → hand → action → smoothing → arm_segmentation → hand_inpaint → robot_inpaint`。`--all-robots` 用于输出所有指定 morphology；默认每个 demo 从 robot_pool 选择单一 morphology。写论文级数据规模时须区别此代码对照和论文作者发布的数据量。

## 复现限制

- Python 3.12+、CUDA 12.6+、PyTorch 2.7+、ffmpeg/ffprobe，另外需手动准备 DA3、SAM 3、ProPainter、AnyHand/WiLoR、Dyn-HaMR 等模型和权重；raw-video 的远程 VLM 条件另需显式 opt-in 和 API key。
- `.env` 数据路径、第三方依赖、Robot XML、ACone 相机/底座标定需要自行准备。RoboTwin 独立环境；仅仓库可读不足以证明论文数值可一键复现。
- **已静态阅读** README、`scripts/run_pipeline.py`、`ego2robot/pipeline.py`、`configs/egoverse_mecka.yaml` 和 `scripts/convert_to_lerobot.py` 关键入口及数据读写；其余 processor 模块通过注册表和结构定位，未逐行审计；未运行 GPU、未下载模型。

## 一手源码参考

- https://github.com/www-Ye/ego2robot_blog/blob/main/README.md
- https://github.com/www-Ye/ego2robot_blog/blob/main/scripts/run_pipeline.py
- https://github.com/www-Ye/ego2robot_blog/blob/main/ego2robot/pipeline.py
- https://github.com/www-Ye/ego2robot_blog/blob/main/configs/egoverse_mecka.yaml
- https://github.com/www-Ye/ego2robot_blog/blob/main/scripts/convert_to_lerobot.py
