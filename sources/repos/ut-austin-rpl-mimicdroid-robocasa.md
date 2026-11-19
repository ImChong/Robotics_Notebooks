# MimicDroid 基准仓库（UT-Austin-RPL/mimicdroid-robocasa）

- **URL：** <https://github.com/UT-Austin-RPL/mimicdroid-robocasa/tree/latest>
- **依赖叉仓库：** `ShahRutav/robosuite`（`abs_robot` 分支）
- **数据集：** <https://huggingface.co/datasets/Rutav/MimicDroidDataset>
- **配套论文：** [arXiv:2509.09769](https://arxiv.org/abs/2509.09769)
- **项目页：** <https://ut-austin-rpl.github.io/MimicDroid/>
- **入库日期：** 2026-11-19
- **核对范围：** 通读 `README.md`、`DATASET.md`、`GETTING_STARTED.md`、`robocasa/scripts/download_mimicdroid_dataset.py`、`robocasa/scripts/replay_dataset.py`、`robocasa/utils/mimicdroid_utils.py` 及 `robocasa/environments/kitchen/play_env/play_env.py` 并核验仓库全目录。
- **通读：** 已完成

## 开源范围与实际内容核查（2026-11-19）

| 模块 | 状态 | 实际入口与功能 |
|------|------|----------------|
| L1–L3 基准环境 | **已开源** | `robocasa/environments/kitchen/play_env/play_env.py`（`BaseEnvForPlay` 及 12 个分级任务如 `PnPSinkToRightCounterPlate` L1/L2/L3，经 `PLAY_TASK_NAME_TO_CLASS` 注册） |
| 数据集下载 | **已开源** | `robocasa/scripts/download_mimicdroid_dataset.py`（从 Hugging Face `Rutav/MimicDroidDataset` 拉取 `training/` 8 GB 玩耍数据与 `TaskDemos/` 250 MB 评估提示） |
| 轨迹回放与环境构造 | **已开源** | `robocasa/scripts/replay_dataset.py` 调用 `robocasa/utils/mimicdroid_utils.py` 中的 `make_env`、`EnvArgs` 与 `reset_to`，支持 `--robots DemoTwoHand` 与 `--robots GR1TwoHand`（`WHOLE_BODY_MINK_IK`） |
| ICL 策略训练与推理 | **未公开发布** | 仓库仅含基准、回放与数据集工具；`replay_dataset.py` 第 19 行保留 `../../../icrt` 相对路径占位，Meta-ICL 检索与策略训练代码未随此仓释放 |

## 交叉

- 论文归档：[`sources/papers/mimicdroid_arxiv_2509_09769.md`](../papers/mimicdroid_arxiv_2509_09769.md)
- 项目页归档：[`sources/sites/mimicdroid-ut-austin.md`](../sites/mimicdroid-ut-austin.md)
- wiki：[`wiki/entities/paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo.md`](../../wiki/entities/paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo.md)
