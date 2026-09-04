# HOST（CGuangyan-BIT/HOST）

- **URL：** <https://github.com/CGuangyan-BIT/HOST>
- **权重：** <https://huggingface.co/Guangyan/HOST>
- **许可：** 仓根截至 2026-09-04 **无** 统一 LICENSE；`policy_training/LICENSE` 继承 Fast-WAM；HF 权重卡 **MIT**
- **语言：** Python 3.10；对齐环境 PyTorch 2.4 + CUDA 12.4；策略环境 PyTorch 2.6 + CUDA 12.4
- **配套论文：** [arXiv:2607.20033](https://arxiv.org/abs/2607.20033)
- **项目页：** <https://host-site.host-robotics.workers.dev/>
- **入库日期：** 2026-09-04

## 状态（2026-09-04）

| 项 | 状态 |
|----|------|
| 对齐 / 耦合 / 策略训练代码 | **已发布** |
| HF 权重 `Guangyan/HOST` | **已发布**（`model.pt`） |
| 论文规模真机数据 | **未发布** |
| 真机闭环评测脚本 | **无**（仅 `eval_openloop.sh`） |
| 内部路径清理 | 见仓内 `OPEN_SOURCE_PATH_TODOS.md` |

## 仓库入口（README）

| 目标 | 路径 / 命令 |
|------|-------------|
| 安装 | 两套 Conda：`alignment/environment.yml` → `HOST_Alignment`；`policy_training/environment.yml` → `HOST_Policy` + `pip install -e ./policy_training --no-deps` |
| 数据契约 | `data_preprocessing/build_task_dictionary.py` → `write_task_paths.py` |
| 对齐训练 | `VIDEO_PATHS=... bash alignment/train_scripts/run_ds.sh`（DeepSpeed ZeRO-3） |
| 进度标签 | `coupling/progress_alignment/build_progress_info.py` |
| 策略骨干 | `policy_training/scripts/preprocess_action_dit_backbone.py`（Wan2.2 → ActionDiT） |
| 策略训练 | `scripts/precompute_text_embeds.py` + `bash scripts/run_train.sh` |
| 开环评测 | `bash scripts/eval_openloop.sh`（录制 episode，不指挥真机） |

论文规模：对齐 10,000 step / 策略 Stage 1 500,000 step / Stage 2 100,000 step，均按 **64 GPU** 报；随仓 launcher 是可改模板，不自动复现该规模。

## wiki

- [`wiki/entities/paper-host-one-shot-human-video.md`](../../wiki/entities/paper-host-one-shot-human-video.md)
