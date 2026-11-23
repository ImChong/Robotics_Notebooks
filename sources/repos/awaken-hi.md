# Awaken-HI 官方 GitHub 仓库（状态归档）

> 一手 repo 来源；截至 2026-11-21 **只有项目介绍和演示素材，并非可运行代码**。

- **类型：** repo
- **官方 URL：** <https://github.com/engineai-robotics/Awaken-HI>
- **项目主页：** <https://engineai-robotics.github.io/Awaken-HI/>
- **论文：** <https://arxiv.org/abs/2611.26420>
- **数据入口：** <https://huggingface.co/datasets/EngineAI-Embodied/Awaken-Human>
- **入库日期：** 2026-11-21
- **核对范围：** GitHub 默认分支 README、根目录文件结构、TODO 清单和作者公开的结果表；没有训练/推理/部署入口可读
- **对应详情：** [Awaken-HI](../../wiki/entities/paper-awaken-hi.md)

## 已发布与待发布

| 资源 | README 状态 | 核对说明 |
|---|---|---|
| Technical report | 已发布 | arXiv v1 HTML / PDF |
| Dataset | README 标已发布 | HF 数据卡仅显示一段样例，完整 50k h 不能据此确认 |
| Code including inference code | 待发布 | 仓库 README 未提供 train/infer API 或可运行模型代码 |
| Model weights | 待发布 | 未提供权重入口 |
| Simulation platform | 待发布 | 未提供可运行仿真平台 |

根目录可见 README.md、docs、preview、.gitignore；示例 GIF 与图像不构成推理实现。**不适用源码运行时序图**，也不应虚构训练命令或重建工程依赖。后续新增官方实现后，需先阅读具体入口，再补真实 sequenceDiagram。

## 结果声明

README 对八项单任务、工业上架、家庭整理和新配对泛化分别报 93.75%、90%、80%、90%；对应协议见[论文归档](../papers/awaken_hi_arxiv_2611_26420.md)。
