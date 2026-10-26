# RoboMIND 2.0: A Multimodal, Bimanual Mobile Manipulation Dataset for Generalizable Embodied Intelligence

> 来源归档

- **类型：** 论文 / 数据集
- **作者：** Chengkai Hou、Kun Wu、Jiaming Liu、Zhengping Che、Di Wu 等
- **机构：** arXiv 页面作者信息；以论文版本所列单位为准
- **发布日期：** 2025-12-31（arXiv:2512.24653）
- **论文：** https://arxiv.org/abs/2512.24653
- **项目页：** https://x-humanoid-robomind.github.io/
- **RoboMIND 2.0 数据集合：** https://modelscope.cn/collections/X-Humanoid/RoboMIND20
- **RoboMIND 1.x 数据卡：** https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND
- **入库日期：** 2026-10-26
- **一句话说明：** 将 RoboMIND 扩展为多模态双臂移动操作数据集，并提出结合高层语义规划与低层视觉语言动作执行的 MIND-2 系统。

## 论文摘录要点

1. **真实数据扩展：** 论文报告超过 310K 条双臂操作轨迹，覆盖 6 种机器人本体和 739 项复杂任务。
2. **新增模态和场景：** 包含约 12K 条触觉增强 episode、20K 条移动操作轨迹，以及另行发布的 20K 条数字孪生仿真轨迹，关注接触丰富和空间范围较大的任务。
3. **MIND-2 分层执行：** 高层 MIND-2-VLM 将自然语言任务拆解为可落地子目标；低层 MIND-2-VLA 结合本体状态生成动作。数据集扩展与配套策略系统是相关但不同的贡献。

## 对 wiki 的映射

- [RoboMIND 主实体](../../wiki/entities/painode-094-robomind.md)：版本演进、数据规模和 MIND-2 系统边界。
- [机器人操作任务](../../wiki/tasks/manipulation.md)：双臂、长程和移动操作任务。
- [VLA 方法](../../wiki/methods/vla.md)：MIND-2-VLA 的策略接口语境。

## 参考链接

- arXiv 摘要：<https://arxiv.org/abs/2512.24653>
- 项目主页：<https://x-humanoid-robomind.github.io/>
- ModelScope 数据集合：<https://modelscope.cn/collections/X-Humanoid/RoboMIND20>
