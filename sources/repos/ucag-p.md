# UCAG-P（Public-BOTs/UCAG-P）

> 来源归档（ingest）

- **标题：** UCAG-P: Unified Camera-Centric Action Geometry Pre-training
- **类型：** repo（项目页托管仓；训练代码待发布）
- **组织：** Public-BOTs
- **代码：** <https://github.com/Public-BOTs/UCAG-P>
- **项目页：** <https://public-bots.github.io/UCAG-P/>
- **论文：** <https://arxiv.org/abs/2608.26058>
- **许可：** 未声明
- **入库日期：** 2026-08-28
- **复核日期：** 2026-09-02
- **一句话说明：** GitHub 目前托管 UCAG-P **项目页与论文配图**；README 明确 *training, inference, and evaluation code will be released soon*，无许可证、无训练脚本、无权重。

## 开源状态（步骤 2.5，2026-09-02）

| 项 | 状态 |
|----|------|
| 仓库内容 | `README.md`、`assets/`（总览/动作空间/架构/数据管线/榜单/真机图）、`web-page/`、`.github` |
| 可运行入口 | **无** — 无 `train.py` / `eval.py` / 配置 / checkpoint |
| 许可证 | **未声明** |
| 描述字段 | *UCAG-P paper figures and project page; code release coming soon* |
| 新闻 | 2026-08-23 仓与项目页；2026-08-26 arXiv:2608.26058 |

判定：**宣称将开源 / 部分占位仓**。wiki 实体页「源码运行时序图」写 **不适用**，待官方放出训练/推理入口后再补。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [UCAG-P 论文实体](../../wiki/entities/paper-ucag-p.md) | 方法、评测与开源边界的读者页 |
| [VLA](../../wiki/methods/vla.md) | 异构数据上的通才操作策略；统一的是 **相机几何** 而非 DiT 原生动作 |
| [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md) | 同小米系 Qwen3-VL-4B 骨干，部署叙事不同 |

## 对 wiki 的映射

- [`wiki/entities/paper-ucag-p.md`](../../wiki/entities/paper-ucag-p.md)
- 交叉 [VLA](../../wiki/methods/vla.md)、[Qwen-RobotManip](../../wiki/entities/qwen-robot-manip.md)、[Qwen-VLA](../../wiki/entities/qwen-vla.md)
