# lucas-maes/le-wm

> 来源归档

- **标题：** LeWorldModel 官方实现
- **类型：** repo
- **代码：** <https://github.com/lucas-maes/le-wm>
- **License：** MIT
- **项目页：** <https://le-wm.github.io/>
- **论文：** <https://arxiv.org/abs/2603.19312>
- **权重 / 数据：** <https://huggingface.co/collections/quentinll/lewm>
- **入库日期：** 2026-09-05
- **一句话说明：** LeWM 官方仓：`jepa.py` + Hydra；依赖 stable-worldmodel / stable-pretraining。

## 开源核查（2026-09-05）

| 项 | 状态 |
|----|------|
| 代码 | **已开源** · MIT · 4391★ |
| 权重 | HF `quentinll/lewm` |
| 数据 | HF HDF5，解压到 `$STABLEWM_HOME` |

## 入口速查

| 命令 | 作用 |
|------|------|
| `uv pip install stable-worldmodel[train,env]` | 环境 |
| `python train.py data=pusht` | 训练 |
| `config/eval/` | CEM 规划配置 |
