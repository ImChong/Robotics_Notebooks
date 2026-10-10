# AlphaStar 官方源码与 StarCraft II 环境

> 官方代码仓核查（复核日期：2026-10-10）

- **AlphaStar package：** <https://github.com/google-deepmind/alphastar>
- **DeepMind 的 PySC2：** <https://github.com/google-deepmind/pysc2>
- **对应 Nature 论文：** <https://doi.org/10.1038/s41586-019-1724-z>
- **官方资料归档：** [AlphaStar 项目资料](../sites/alphastar-deepmind.md)
- **关联开源论文：** [StarCraft II Unplugged（NeurIPS 2021）](https://openreview.net/pdf?id=Np8Pumfoty)

## 仓库实际提供什么

| 仓库部分 | README 描述 | 可复现范围 |
|----------|-------------|------------|
| `architectures/` | 面向 StarCraft II agent 的通用网络架构 | 可配合 online/offline learning algorithms 使用 |
| `unplugged/` | 数据读取、offline training 与 evaluation | 目前公开示例算法为 Behavior Cloning（BC）；可研究全离线学习 |
| PySC2 converters | StarCraft II 观测/动作接口及转换工具 | 环境接入、数据转换和评测基础设施 |

README 明确说明：**此仓库未提供 online RL training code**。因此它不是 2019 年 AlphaStar 的完整训练配方，也不应被描述为发布了原始 Grandmaster 权重。AlphaStar Unplugged 论文将其用于离线 RL benchmark、行为克隆基线和基于人类回放的数据研究。

## 运行入口与边界

- 环境：Linux、Python 3.9；可用 `pip install -e .` 或 Bazel 构建。
- 离线训练入口：`python alphastar/unplugged/scripts/train.py --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.full ...`
- 评估入口：`python alphastar/unplugged/scripts/evaluate.py ...`
- 真实数据运行需先按 `alphastar/unplugged/data/README.md` 生成/准备数据，并配置 replay dataset 路径。README 中的 dummy quickstart 只验证训练框架，不是完整 benchmark 实验。
- Nature 2019 在线 League 的数据生成、训练和最终智能体权重不包含在该仓库公开范围内。

## 对 wiki 的映射

- [AlphaStar 独立项目节点](../../wiki/entities/paper-alphastar.md)
- [DeepMind 官方研究资料](../sites/alphastar-deepmind.md)
