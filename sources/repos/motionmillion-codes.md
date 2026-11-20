# MotionMillion-Codes（VankouF/MotionMillion-Codes）

> 来源归档

- **类型：** repo
- **官方仓库：** <https://github.com/VankouF/MotionMillion-Codes>
- **项目页：** <https://vankouf.github.io/MotionMillion/>
- **论文：** <https://arxiv.org/abs/2507.07095>
- **数据集：** <https://huggingface.co/datasets/InternRobotics/MotionMillion>
- **许可：** Apache-2.0
- **入库日期：** 2026-07-28
- **通读：** 已完成
- **核对范围：** 2026-11-20 通读 `VankouF/MotionMillion-Codes` 官方 `README.md`（295 行全文）及推理入口 `inference_single.py`（L1–100、L271–370），核实可选 prompt 改写、CLIP/Flan-T5-XL/XXL 编码、`LLaMAHF` 自回归采样、`net.forward_decoder` 解码、`vector_272` 逆变换及 `recover_from_local_rotation` 运行流程。
- **开源状态：** **已开源** — 训练、单条/批量推理、评测、3B/7B checkpoint 下载与脚滑后处理代码已发布。

## 可运行入口（2026-07-28）

| 任务 | README 入口 |
|------|-------------|
| 下载权重 | `bash prepare/download_pretrained_models.sh` |
| 3B/7B 单条推理 | `scripts/inference/single_inference/test_t2m_{3B,7B}.sh` |
| MotionMillion-Eval | `scripts/inference/batch_inference/test_t2m_{3B,7B}.sh` |
| FSQ tokenizer 训练 | `scripts/train/train_tokenizer*.sh` |
| 文本到动作训练 | `scripts/train/train_t2m_{3B,7B}.sh` |
| 评测 | `scripts/eval/eval_tokenizer.sh`、`eval_t2m_{3B,7B}.sh` |
| 脚滑后处理 | `postprocess/remove_sliding/scripts/run_remove_sliding.sh` |

README 基线为 Python 3.8.11、PyTorch 2.4.1；SMPL+H、DMPL、T5-XL 与大模型权重需另行下载，7B 训练不是单卡轻量复现。

## 对 wiki 的映射

- [`paper-notebook-go-to-zero-towards-zero-shot-motion-generation-w.md`](../../wiki/entities/paper-notebook-go-to-zero-towards-zero-shot-motion-generation-w.md)
- [`motionmillion-project.md`](../sites/motionmillion-project.md)
- [`humanoid_pnb_go-to-zero.md`](../papers/humanoid_pnb_go-to-zero.md)
