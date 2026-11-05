# BlinkMimic

> 来源归档（社区实现）

- **链接：** <https://github.com/fuyichen0725/blinkmimic>
- **核查版本：** [d264658](https://github.com/fuyichen0725/blinkmimic/tree/d264658d6c9ce61fb52f9f0f8985a369f3ccc409)
- **类型：** repo；基于 InstantTrack 和 supermimic_vibed 的社区实现，并非 SuperMimic 官方代码。
- **许可证：** 代码 MIT；LAFAN1 动作和数据衍生的 checkpoint 等另受非商业许可约束。
- **入库日期：** 2026-11-03
- **沉淀到 wiki：** [BlinkMimic](../../wiki/entities/blinkmimic.md)

## 核查范围

通读该版本 README，核对 scripts/train.sh、scripts/presets.sh 与训练主循环、环境和模型源码的相关部分。数据下载自 lvhaidong/LAFAN1_Retargeting_Dataset；训练与评测入口分别为 `bash scripts/train.sh logs/run_s0 --seed 0` 与 `uv run python tools/eval_batch.py --run_dir logs/run_s0 --group 18 --reps 4`。尚未运行训练或独立复现。

README 报告 RTX 4090 同机 40 段 LAFAN1：严格目标下 BlinkMimic 三种子平均 3.38 分钟，InstantTrack v1.2 为 7.36 分钟；完整定义和计时边界见实体页。SuperMimic 两组时间是按迭代数和实测单步时间估算，不是同一完整计时实验。
