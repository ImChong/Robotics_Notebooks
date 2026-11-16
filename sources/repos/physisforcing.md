# DAGroup-PKU/PhysisForcing

- **类型：** repo
- **官方入口：** <https://github.com/DAGroup-PKU/PhysisForcing>
- **版本：** `6b407f5776479d17fa3574c3697e42e738b2ba0c`（2026-11-13 核对）
- **论文：** [arXiv:2606.28128](../papers/physisforcing_arxiv_2606_28128.md)
- **项目页：** <https://dagroup-pku.github.io/PhysisForcing.github.io/>
- **权重：** <https://huggingface.co/DAGroup-PKU/PF_Wan>、<https://huggingface.co/DAGroup-PKU/PF_Cosmos>、<https://huggingface.co/DAGroup-PKU/PhysisForcing-assets>
- **开放结论：** 已发布 PF_Wan / PF_Cosmos 推理、训练代码和示例；各自依赖环境不同。
- **核对范围：** 根 README、PF_Wan README、`tools/main.py` / `launch_1node.sh`、full-ft 配置；`DiffusionFinetuning` 的初始化、run/training_loop/run_step 与 token mask；projection/hook 与 relational loss。只做静态阅读，未运行 GPU 实验；未逐行复核 PF_Cosmos。
- **通读：** 已完成

## 实际运行路径（PF_Wan）

1. `pf_wan/tools/launch_1node.sh` 检查 `weights/`，torchrun 启动 `tools/main.py`。
2. `tools/main.py` 合并 JSONC 与 CLI 配置，经 `ENGINE_REGISTRY` 建 `DiffusionFinetuning`，调用 `engine.run()`。
3. `DiffusionFinetuning.run` 从 dataloader 取 batch，prepare_inputs 后 `training_loop` 调 `run_step`；其中 hook DiT 中间层，低层 head + track_loss 对齐轨迹，高层 head 对齐 V-JEPA2 网格与选定 token 的关系矩阵。
4. `run_step` 返回三 loss 之和；`training_loop` backward，`run` 按 accum_steps 调 optimize 与 save。

复现注意：full-ft 示例配置 `training.steps=1000000`，论文 Wan 实验为 20k；不要把当前配置默认值直接当论文超参。示例块索引 20，论文表 6 的另一消融最佳层为 15。

## 对 wiki 的映射

- [PhysisForcing](../../wiki/entities/paper-physisforcing.md)
