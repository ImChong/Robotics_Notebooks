# HOST（CGuangyan-BIT/HOST）

- **URL：** <https://github.com/CGuangyan-BIT/HOST>
- **权重：** <https://huggingface.co/Guangyan/HOST>
- **许可：** 仓根截至 2026-11-19 **无** 统一 LICENSE；`policy_training/LICENSE` 继承 Fast-WAM；HF 权重卡 **MIT**
- **语言：** Python 3.10；对齐环境 PyTorch 2.4 + CUDA 12.4；策略环境 PyTorch 2.6 + CUDA 12.4
- **配套论文：** [arXiv:2607.20033](https://arxiv.org/abs/2607.20033)
- **项目页：** <https://host-site.host-robotics.workers.dev/>
- **入库日期：** 2026-09-04
- **更新日期：** 2026-11-19
- **核对范围：** 通读 `README.md`、`OPEN_SOURCE_PATH_TODOS.md`、`alignment/`、`coupling/`、`policy_training/README.md`、训练与开环评测源码（`alignment/train.py`、`alignment/algos/alignment.py`、`coupling/progress_alignment/build_progress_info.py`、`policy_training/scripts/train.py`、`runtime.py`、`self_grounded_predictor_joint.py`、`evaluate_openloop.py`、`self_grounded_predictor_eval.py`）。
- **通读：** 已完成

## 状态（2026-11-19）

| 项 | 状态 |
|----|------|
| 对齐 / 耦合 / 策略训练代码 | **已发布** |
| HF 权重 `Guangyan/HOST` | **已发布**（`model.pt` + `config.yaml`） |
| 论文规模真机数据 | **未发布**（193,462 条轨迹与 5,847 条人视频配对不在仓内） |
| 真机闭环评测脚本 | **无**（`eval/real_openloop/` 仅对录制 episode 做开环预测对比） |
| 内部路径清理 | 见仓内 `OPEN_SOURCE_PATH_TODOS.md`（配置仍留内部集群路径占位） |

## 仓库入口与核心类（源码核对）

| 目标 | 路径 / 类或入口 | 说明 |
|------|-----------------|------|
| 安装 | `alignment/environment.yml` → `HOST_Alignment`；`policy_training/environment.yml` → `HOST_Policy` + `pip install -e ./policy_training --no-deps` | 对齐与策略使用两套独立 Conda 环境 |
| 数据契约 | `data_preprocessing/build_task_dictionary.py` → `write_task_paths.py` | 按任务分组生成对齐所需的 `task_paths.json` |
| 对齐训练 | `alignment/train.py` → `algos/alignment.py` (`Alignment` + `compute_deterministic_alignment_loss_paired` / `smooth_dtw.py`) | DeepSpeed ZeRO-3 训练 Qwen3-VL-Embedding-8B，输出 `high_loss_samples_*.jsonl` |
| 进度标签 | `coupling/progress_alignment/build_progress_info.py` (`process_jsonl`) | 读取对齐 JSONL，为每个 episode 写入任务进度标签 `info_dtw.json` |
| 策略骨干准备 | `policy_training/scripts/preprocess_action_dit_backbone.py` | 从 Wan2.2 DiT 线性插值构建 `ActionDiT` 初始权重 |
| 策略训练 | `policy_training/scripts/train.py` → `runtime.py::create_self_grounded_predictor` → `SelfGroundedPredictorJoint` | MoT（Wan2.2-TI2V-5B 视频专家 + ActionDiT 动作专家，Action attend ALL video tokens）；由 `scripts/run_train.sh` 启动 |
| 开环评测 | `policy_training/eval/real_openloop/evaluate_openloop.py` → `SelfGroundedPredictorEval` (`set_task_video` + `infer_real` → `infer_joint`) | 加载 checkpoint 对录制轨迹做 chunk 级动作 MAE/RMSE 评测，不连接真机控制 |

论文规模：对齐 10,000 step / 策略 Stage 1 500,000 step / Stage 2 100,000 step，均按 **64 GPU** 报；随仓 launcher 是可改模板，不自动复现该规模。

## wiki

- [`wiki/entities/paper-host-one-shot-human-video.md`](../../wiki/entities/paper-host-one-shot-human-video.md)
