# Tactile-based Multimodal Fusion — 论文归档

- **标题：** Tactile-based Multimodal Fusion in Embodied Intelligence: A Survey of Vision, Language, and Contact-Driven Paradigms
- **类型：** paper / survey
- **版本：** arXiv:2605.17336v1，2026-05-17
- **论文：** https://arxiv.org/abs/2605.17336v1
- **全文：** https://arxiv.org/html/2605.17336v1
- **资源仓库：** https://github.com/Wayne-coding/Multimodal-Tactile-Fusion
- **作者：** Zhixiang Cao, Di Tian, Runwei Guan, Yanzhou Mu, Xiaolou Sun, Shaofeng Liang, Daizong Liu, Tao Huang, Yutao Yue, Henghui Ding, Bin Fang, Alex Zhou, Qing-Long Han, Hui Xiong
- **核对范围：** v1 HTML 全文 §I–VIII、公式 (1)–(4)、表 I–III 与图 1–8 的说明；涵盖数据、方法、硬件、评测、讨论与结论。全文未另列附录。官方 README 与根目录核查于 2026-11-07。
- **通读：** 已完成

## 核心摘录

1. §II 将融合抽象为物理换能及观测、各模态编码、联合表示、任务解码四步。拼接、交叉注意力和对比对齐是不同选择，并非每个方法都同时使用。
2. §III / 表 I 区分 T-V、T-L、T-V-L、T-V-O；O 可包括动作、音频和本体感受。ImageNet-T 没有真实触觉测量，不能与物理传感数据等同。
3. §IV / 表 II 分为感知识别、跨模态生成、交互操作；语言语义对齐不等于接触控制能力。
4. §V 的硬件分类包括可穿戴、手持/指尖、机器人皮肤/多模态贴片、夹爪集成；形态影响观测与迁移。
5. §VI / 表 III 汇总各文献评测，未提供统一重跑的排行榜。识别、检索、生成和操作指标测量不同能力；跨传感器与跨数据集仍有域偏移。
6. §VII 主张时间对齐、动作条件轨迹、分层融合、不确定性与顺应性约束，以及统一触觉 token 和接触仿真；这些是研究方向，不能当作已验证方案。

## 开放状态

官方公开的是文献与资源目录，根目录仅 README、imgs 与 .gitignore；未见本综述自有训练/推理实现、权重或新数据下载。链接到其他工作的代码不等于综述提供统一可运行框架。README 作者栏与论文作者存在差异，书目作者以 v1 论文为准。

## 对 wiki 的映射

- [触觉多模态融合综述](../../wiki/entities/paper-tactile-multimodal-fusion-survey.md)
- [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)
- [官方资源库核查](../repos/multimodal-tactile-fusion.md)
