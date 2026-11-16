# S-EMBER: A Large-Scale Benchmark for Streaming Egocentric Memory Retrieval｜流式第一视角情景记忆基准

- **类型：** paper，arXiv:2607.02689（2026）
- **作者：** Xiaodong Wang, Xuanyi Zhao, Pedro Rodriguez, et al.
- **机构：** Meta FAIR / Meta Reality Labs
- **论文：** https://arxiv.org/abs/2607.02689
- **官方代码：** https://github.com/facebookresearch/S-EMBER
- **数据：** https://huggingface.co/datasets/facebook/S-EMBER
- **核对范围：** arXiv:2607.02689v2（2026-08-26）正文第 1–8 节及附录 A–H（含原 PDF 附录 H 提示词，HTML 转换未正确显示），表 1–5、注释规范、时间采样和判分指标；官方 README、data/README、评测脚本与 utils.py。
- **通读：** 已完成
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

- [x] 已核读 arXiv v2 全文及附录 A–H；附录 H 在 HTML 版本显示转换错误，实际以 [17 页 PDF](https://arxiv.org/pdf/2607.02689) 第 16–17 页为准。

## 正文、附录与代码交叉核验（2026-11-16）

- **官方全称：** 论文摘要定义 **S-EMBER = Streaming Egocentric Memory Benchmark for Episodic Retrieval**；**GSER = Grounded Streaming Episodic Retrieval**，两者不可混淆。
- **采集与注释：** §3/附录 G 报告 613 名佩戴者、3,141 段智能眼镜视频、388 小时、9,448 QA；8 类问题、回答短/中/长三层，以及证据时间区间。音频在问答标注过程中静音/不纳入评测；附录 G 限制视频开头 **1 分钟**不能提问，且问题须覆盖至少三种类型。证据额外 padding 不得超过证据长度的 **15%** 或 **15 秒**，取较小者。
- **指标定义：** §4 把语义回答 `Acc_ov` / `Acc_cl` / 5-way MCQ 与证据定位 `mIoU` / `R@1(IoU≥τ)` 分开；真正的联合指标是 **GQ@τ**，同时要求答案语义正确、证据区间达到阈值。代码当前 grounding 默认路径仅汇总 temporal IoU，与论文完整 GQ@τ 定义不同。
- **量化结果：** §5 表 3，Gemini 3.1 Pro 的 MCQ **54.41%**、mIoU **0.479**、GQ@0.5 **0.297**，人工参考 GQ@0.5 **0.698**；开源 Qwen3VL-32B 的 GQ@0.5 **0.163**（同表）。这些是论文报告，不是本次复现得分。
- **诊断和消融：** §5.3 指出长时记忆衰减；§6 发现 128→512 帧仅增加约 **3.2** mIoU，360p 后提高分辨率收益很小。§7/附录 F 的两遍自适应索引先用 64 帧预测证据区间，再以 64 均匀+64 局部帧重新读取；表 4 的 GQ@0.5 有提升但不能达人工水平。
- **协议和局限：** 附录 D：开放模型共同使用提问前 **128 帧**、绝对时间戳，原视频存储 720p/12fps；闭源模型采用不同的接口与取帧预算，**不可直接作严格受控横向比较**。附录 B 的 600 对人工验证 judge accuracy **96.2%**；附录 C 定义 hallucination 判据；附录 E 记录 H200 计算条件；PDF 附录 H 定义了 `Answer:` 与 `Time: [start,end]` 格式和 JSON 判分要求。
- **复现边界：** 已核对官方评测源码入口与主要处理逻辑；未下载全部 388 小时素材，也未复现模型、Gemini judge 或完整 GQ 指标。
