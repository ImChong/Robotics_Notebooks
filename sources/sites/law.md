# LAW 项目页

- **标题：** LAW: Unifying Language, Action, and World Modeling for Agentic Physical Intelligence
- **类型：** site / project-page
- **URL：** <https://hey-cjj.github.io/LAW-project/>
- **论文：** <https://arxiv.org/abs/2610.23334>（2026-10-16）
- **代码：** <https://github.com/hey-cjj/LAW>（项目论文及官方 README 链接）
- **模型：** LAW-FastWAM、LAW-VLM-FastWAM、LAW-OpenWAM、LAW-VLM-OpenWAM（Hugging Face 链接见[官方 README](https://github.com/hey-cjj/LAW#2-models-and-data)）
- **训练数据：** LAW-RoboTwin-WAM-Training、LAW-RoboTwin-VLM-Training（公开于 Hugging Face，见官方 README）
- **核查日期：** 2026-10-20
- **一句话说明：** 论文、项目页和官方仓库围绕同一 LAW 闭环 VLM–WAM 系统；以语义子任务进度连接长时程规划、动作选择与完成检测。

## 开源范围

**已开源。** 官方 GitHub README 列出模型检查点、训练初始化权重、WAM/VLM 训练数据、RoboTwin 资产以及下载、校验、训练、单任务推理和 50 项任务评测入口。代码仓根目录代码采用 MIT；作者训练权重采用 Apache-2.0；标注采用 CC-BY-4.0；第三方上游依赖、媒体和资产另按各自许可执行，使用前须检查仓库 NOTICE 与许可范围说明。

## 复现提示

训练与完整评测说明要求 Linux、Python 3.10 和 NVIDIA GPU；完整训练/评测使用 4 节点 × 8 GPU。WAM 训练前需预计算视频 latent；FastWAM 与 OpenWAM 使用不同 profile / 环境。具体命令及资产下载步骤见[官方 README](https://github.com/hey-cjj/LAW)。

- **归档：** [代码仓归档](../repos/law.md) · [论文摘录](../papers/law_arxiv_2610_23334.md)
- **wiki：** [LAW 实体页](../../wiki/entities/paper-law.md)
