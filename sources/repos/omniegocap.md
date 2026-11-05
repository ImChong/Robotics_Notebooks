# OmniEgoCap 官方代码

- **类型：** repo
- **代码：** <https://github.com/KyungWonCho/OmniEgoCap>
- **项目页：** <https://kyungwoncho.github.io/OmniEgoCap/>
- **论文：** <https://arxiv.org/abs/2512.19283>
- **入库日期：** 2026-11-04

官方 README 标注 2026-10-19 发布代码及 checkpoint。代码 MIT；checkpoint CC BY-NC 4.0，人体模型与数据另有许可。

| 入口 | 作用 |
|------|------|
| `scripts/prepare_input.py` | 调用 Aria MPS 或 trajectory adapter 生成 EgoSequence NPZ |
| `scripts/sample.py` | 加载 checkpoint 和 EgoSequence，调用 predict，保存 SMPL-H NPZ；默认启用 guidance optimization |
| `scripts/visualize.py` | 加载输入、预测及 SMPL-H 模型，启动 viewer |
| `scripts/train.py` / `scripts/eval.py` | 训练与评测入口（依据 README） |

核对范围：README 与上述三个推理入口；尚未逐行审核模型实现或实际运行。

- [论文归档](../papers/omniegocap-2512-19283.md)
- [项目页归档](../sites/omniegocap.md)
- [实体页](../../wiki/entities/paper-sa-2512-19283-omniegocap-camera-agnostic-sequence-level-egocen.md)
