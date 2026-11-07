# XMimic / JAX 世界模型动作跟踪训练器

- **链接：** <https://github.com/nico-bohlinger/xmimic>
- **版本：** `571715f02c51654dca07b5dfa6651d0e586ca93e`，2026-11-07
- **类型：** repo
- **论文：** <https://github.com/nico-bohlinger/papers/blob/main/2026_xmimic.pdf>
- **源码状态：** 已开源；训练、评测、CPU 回放、数据下载与评测 CSV 可获取。
- **许可证：** MIT；LAFAN1 数据及衍生物 CC BY-NC-ND 4.0，非商业。
- **核对范围：** README 全文、train.py、single_gpu.py、multi_gpu.py（含 AWLP 的 _run_awl、Prober）、algorithm.py 采集与世界模型更新、fast.py 策略/世界模型更新、evaluate.py、results/README.md 与 make_figures.py 统计规则；未运行训练。
- **通读：** 已完成
- **沉淀到 wiki：** [XMimic](../../wiki/entities/paper-xmimic-jax.md)

## 入口与证据

`python train.py --layout AWLP --log_dir runs/awlp` → `multi_gpu.run` → `_run_awl`，异步 actor 线程、WM/critic 与策略程序交换上一轮结果，独立 Prober 回传失败分布。默认单卡走 `single_gpu.run`。`python -m xmimic.evaluate runs/awlp` 加载 NPZ checkpoint 并输出 CSV；`tools/make_figures.py` 聚合 results/runs.csv 列出的逐运行结果。

当前 README 主表仅直接比较 BlinkMimic、InstantTrack v1.3、XMimic。四卡 3.5× 是墙钟对比，单卡约 1.29×；首次编译 2–3 分钟不计入训练时钟。[论文归档](../papers/xmimic-jax.md)。与 Open-X-Humanoid/xMimic 无关。
