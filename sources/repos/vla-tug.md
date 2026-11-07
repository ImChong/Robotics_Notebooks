# VLA-Tug 官方匿名源码与数据

- **仓库：** <https://anonymous.4open.science/r/VLA-Tug/>
- **项目页：** <https://vla-tug.github.io/>
- **论文：** <https://arxiv.org/abs/2611.08425>
- **核查日期：** 2026-11-07；镜像显示 2026-11-04 snapshot、172 文件、可用至 2027-03-06；未暴露可核实 Git commit SHA，不伪造版本。
- **代码状态：** 已公开（匿名镜像）；MIT，数据/文档与结果 CC BY 4.0；移植代码许可证见 NOTICE。
- **通读：** 已完成
- **核对范围：** README、data/README.md、cli/download.py、cli/train.py、cli/evaluate.py、fp/config.py、fp/data/download.py、fp/models/ours.py、fp/train/sweep.py、fp/train/ours.py、fp/eval/run_eval.py；核查运行入口、主模型、训练与评测调用，未声称全库审计或实际训练复现。

## 实际运行链路

- `python -m cli.download liberox_groot` → `fp.data.download.download_dataset` → Hub `snapshot_download` → manifest、split 与 pooled 特征/targets 移至 `data/<dataset>/`；不下载原始 rollout。
- `python -m cli.train --datasets liberox_groot --variants onset_pre --probes lstm_modes_clip` → `load_inputs` / preflight → `train_unit` → 仅训练 split 的特征统计量标准化 → `OursScorer.fit` → `Probe` / 训练循环 → 验证评分损失选 checkpoint → `outputs` 给全部 episode 打分 → `write_run` 保存结果。
- `python -m cli.evaluate --datasets liberox_groot` → `evaluate_saved_run` → `evaluate_run` / `evaluate_scores`：读取保存分数，在校准成功轨迹拟合 max-conformal，评测 val/seen/unseen 并输出 JSON；此命令不调用 VLA 或仿真器。
- `--ltt --alpha-tpr-early 0.3 --alpha-fpr 0.2 --delta 0.1` 另走 `evaluate_ltt` / `calibrate_ltt`，用成功与失败校准轨迹；目标与阈值不等同普通 sweep，正文表格复现用 `cli.ltt_table --delta 0.1 --n-starts 40`。

## 数据资源

源码 `fp/config.py` 注册以下数据；2026-11-07 Hub API 均返回公开 repo id 与 pooled 文件清单，仅验证可发现性与文件存在，未校验大数组内容。

| 数据 | 官方入口 |
|---|---|
| LIBERO-10 GR00T | <https://huggingface.co/datasets/podolinsky/groot-libero-10-features-v3> |
| LIBERO-10 π0.5 | <https://huggingface.co/datasets/podolinsky/pi0.5-libero-10-features-v3> |
| LIBERO-X GR00T | <https://huggingface.co/datasets/podolinsky/gr00t-libero-x-features-v3> |

真机 `real8_groot` 的 hf_repo 为空，README 明确不发布。动作一致性基线需另外准备动作块；基础 download 不取这些额外数据。源码是离线探针复现链路，不能把论文机器人 IO 演示当成现成部署 API。

## 对 wiki 的映射

- [主节点](../../wiki/entities/paper-vla-tug.md)
- [论文归档](../papers/vla-tug.md)
- [项目页核查](../sites/vla-tug.md)
