# danijar/dreamerv3

> 来源归档

- **标题：** DreamerV3 开源实现（danijar）
- **类型：** repo
- **组织 / 作者：** Danijar Hafner
- **代码：** <https://github.com/danijar/dreamerv3>
- **项目页：** <https://danijar.com/dreamerv3>
- **论文：** <https://arxiv.org/abs/2301.04104>（Nature 版叙事见仓内 BibTeX）
- **License：** MIT
- **入库日期：** 2026-07-27
- **一句话说明：** DreamerV3 的公开 JAX 复现：世界模型 + 想象轨迹上的 actor-critic；固定超参跨域；入口 `python dreamerv3/main.py --configs …`。README 声明基于 DreamerV2 开源代码库的 reimplementation，与 Google/DeepMind 内部实现无关。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `pip install -U -r requirements.txt`（先装 JAX） | 依赖 |
| `python dreamerv3/main.py --logdir … --configs crafter --run.train_ratio 32` | 训练 |
| `--configs atari --task atari_pong` 等 | 复现对应域 |
| `dreamerv3/configs.yaml` | 全部配置项 |
| `python -m scope.viewer --basedir ~/logdir --port 8000` | 可视化 |
| Docker | 仓内 `Dockerfile` |

## 开源状态（仓库核查，2026-07-27）

| 资产 | 状态 |
|------|------|
| 训练代码 | **已开源** · MIT |
| 与 Nature/DeepMind 官方内部实现 | README：**unrelated to Google or DeepMind** |
| 后继 | Dreamer 4 / [Open Dreamer](open-dreamer.md) 为更新一代管线 |

## 对 wiki 的映射

- 论文策展：[`sources/papers/shenlan_wm_survey_13_dreamerv3.md`](../papers/shenlan_wm_survey_13_dreamerv3.md)
- 沉淀 **[`wiki/entities/paper-shenlan-wm-13-dreamerv3.md`](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md)**
- 后继复现：[`sources/repos/open-dreamer.md`](open-dreamer.md)
