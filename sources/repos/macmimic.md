# MacMimic | 作者仓库与实验记录

- **类型：** repo
- **作者：** Jian Zhang（GitHub: air23zj）
- **代码：** <https://github.com/air23zj/macmimic>
- **固定版本：** [d9a9c745](https://github.com/air23zj/macmimic/tree/d9a9c745d1be5179ddb46dda424be3d819bf4823)
- **项目页：** <https://air23zj.github.io/macmimic/>
- **许可：** 代码 MIT；动作数据保留其各自许可。
- **核对范围：** 作者项目页、README、RESULTS；固定版本中的训练 shell、采集与 learner 同步分支、训练 checkpoint 及阈值脚本。
- **通读：** 已完成
- **对应页面：** [MacMimic](../../wiki/entities/macmimic.md)

作者在 M4 Pro 48 GB MacBook Pro 上，用同一批重定向到模拟 Unitree G1 的 40 个 LAFAN1 片段训练和评测。在相隔 100 iterations 的连续三个检查点满足归一化 return ≥0.82 且无失败时，记录**第一个**检查点的训练耗时。三个种子为 21.0、21.3、22.5 分钟。对照为本仓在同一硬件的 BlinkMimic MLX 移植单次运行，44 分钟；不能解释为超越原 CUDA 版或等质量加速。

## 代码入口

- `scripts/train.sh` 调 `python -m supermimic_mlx.train`；overlap 分支通过 `LearnerThread` 排队，在 actor 快照采集与 GPU 学习并行后等待 learner、提交延后的归一化更新。
- `supermimic_mlx.evaluate` 逐 checkpoint 仿真评测，`tools/time_to_target.py` 读取 `eval.csv` 与 `args.json` 的快照步距，寻找连续达标序列。
