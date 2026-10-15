# fast-feature-fields

> 官方代码仓库归档

- **类型：** repo
- **仓库：** <https://github.com/grasp-lyrl/fast-feature-fields>
- **论文：** [arXiv:2509.25146](https://arxiv.org/abs/2509.25146)；期刊版 [DOI 10.1126/scirobotics.aec8218](https://doi.org/10.1126/scirobotics.aec8218)
- **项目页：** <https://www.seas.upenn.edu/~richeek/f3/>
- **机构：** University of Pennsylvania, GRASP Laboratory
- **入库日期：** 2026-10-15
- **一句话说明：** F³ 官方实现，包含事件表示训练、机器人感知下游任务、预训练推理和 PyTorch 2 AOTI 导出代码。
- **代码状态：** 已开源；训练数据按 data/README.md 准备。

## 主要入口

| 仓库入口 | 用途 |
|---|---|
| minimal.ipynb / hubconf.py | 预训练推理示例与 PyTorch Hub 加载 |
| main.py / src/f3/event_FF.py | F³ 表示训练入口与核心模型 |
| src/f3/tasks/depth/ | 单目深度任务 |
| src/f3/tasks/optical_flow/ | 光流任务 |
| src/f3/tasks/segmentation/ | 语义分割任务 |
| src/f3/tasks/matching/ | 匹配相关工具 |
| _aoti_pt2/ | PyTorch 2 AOTI 导出与 Python / C++ 推理 |
| data/README.md | M3ED、DSEC、MVSEC 等数据配置说明 |

- 主实体：[Fast Feature Field (F³)](../../wiki/entities/paper-fast-feature-field-f3.md)
- 项目页：[F³ 项目页归档](../sites/fast-feature-field-f3.md)
