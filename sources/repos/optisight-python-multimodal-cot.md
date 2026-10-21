# OptiSight 作者代码仓库

> 来源归档（repo）— 核对公开仓库 README；代码结构与依赖以仓库当前版本为准。

- **标题：** OptiSight CoT Autonomous Navigating System
- **类型：** repo
- **链接：** https://github.com/avanalperen/OptiSight-Python-Multimodal-CoT-for-Visual-Reasoning
- **论文：** https://arxiv.org/abs/2608.23354
- **IRI 官方记录：** https://www.iri.upc.edu/publications/show/3160
- **入库日期：** 2026-10-20
- **一句话说明：** Python 仓库包含本地 dashboard、AI Habitat 交互和 OptiSight CoT 导航模式；`start.py` 检查资源并启动服务。

## 开源状态与复现边界

- **公开代码：** 仓库可公开访问，提供 `start.py`、`codes/`、`setup/` 等系统入口与运行说明。
- **外部资源：** Matterport/HM3D 场景地图与分割、检测、VLM 模型文件需用户自行下载；README 声明大型地图和模型权重未上传。
- **评测：** README 声明仓库没有自动计算论文成功率的评估脚本；其 24 个场景各重复 10 次的汇总结果由人工测试和统计。
- **硬件：** README 建议配备 CUDA NVIDIA GPU；仓库不能单独证明论文所报 8 GB 显存预算适用于任意 GPU 与配置。
- **沉淀到 wiki：** [OptiSight 论文实体页](../../wiki/entities/paper-optisight-embodied-navigation.md)
