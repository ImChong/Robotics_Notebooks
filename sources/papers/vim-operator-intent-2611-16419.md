# I See What You Want: Decoding Hierarchical Operator Intent from Visual Teleoperation History｜VIM：视觉历史推断遥操作层级意图

- **类型：** 研究原始论文
- **论文：** https://arxiv.org/abs/2611.16419 （v1）
- **HTML：** https://arxiv.org/html/2611.16419v1
- **PDF：** https://arxiv.org/pdf/2611.16419
- **项目：** https://see-want.github.io/
- **代码：** https://github.com/see-want/VIM
- **核对范围：** arXiv 原始版本的研究任务、核心方法与算法、主要实验评测、结论及限制；不将第三方摘要作为通读依据。没有在本次执行模型训练或真机复现。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [论文详情](../../wiki/entities/paper-vim-operator-intent-2611-16419.md)

## 方法与评测核对

- **研究问题：** 用短时遥操作视觉历史同时预测当前原子动作 OA 和上层任务 OM，支持人机接力。
- **方法原理：** 对跨来源遥操作片段标注动作与任务两个层次及时间边界；Qwen3.5-27B 加 LoRA 利用四帧多视角历史与已完成动作预测 OA/OM；真机由人确认所推断意图，再交给已微调 VLA 执行。
- **原文结果：** VI AMO 6600 段 episode、13,653 个动作、9702 个 mission，VIM 在行动与任务语义指标上高于零样本对照；真机测试为有人工确认的接管。
- **局限与范围：** 官方 VI AMO 数据需获得访问权限，模型训练需较大显存；不能把人工确认加微调的接管称为自主零样本接管。
