# S-EMBER: A Large-Scale Benchmark for Streaming Egocentric Memory Retrieval｜流式第一视角情景记忆基准

- **类型：** paper，arXiv:2607.02689（2026）
- **作者：** Xiaodong Wang, Xuanyi Zhao, Pedro Rodriguez, et al.
- **机构：** Meta FAIR / Meta Reality Labs
- **论文：** https://arxiv.org/abs/2607.02689
- **官方代码：** https://github.com/facebookresearch/S-EMBER
- **数据：** https://huggingface.co/datasets/facebook/S-EMBER
- **核对范围：** 论文 GSER 任务设定、数据八类记忆题和基准结果；官方 README、data/README、评测脚本与 utils.py；附录尚未全部逐段复核。
- **通读：** 未完成
- **入库日期：** 2026-11-16
- **wiki：** [论文实体](../../wiki/entities/paper-s-ember.md)

## 原文信息与 Wiki 映射

1. **GSER（Grounded Streaming Episodic Retrieval）：** 每个问题给定提问时刻，只能读取之前的历史画面，同时需要回答和定位支持回答的时间片段。
   - **对 wiki 的映射：** 以 query_time 截断历史、分别对答案与定位做评测。
2. **真实日常第一视角：** 官方发布 3,141 个视频、388 小时、9,448 组 QA；涵盖位置轨迹、顺序动作、计数、视觉细节、时间排序、持续时间、物体对比和空间推理。
   - **对 wiki 的映射：** 区分第一视角记忆基准与一般在线时空理解基准。
3. **评测协议：** MCQ 计算答案准确率；grounding 产生自由回答和时间区间，默认工具输出 temporal IoU，语义判分需另外运行可选 judge。
   - **对 wiki 的映射：** 不把纯时间 IoU 错当成语义/时间联合得分。
4. **代码证据：** utils.py 通过 video_end=question_time 控制输入视频范围，patched 模型 wrappers 实际裁剪。
   - **对 wiki 的映射：** 源码运行时序图。

- [ ] 完成全篇及附录逐段核验后更新“通读”。
