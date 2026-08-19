# Motion-as-Prompt（arXiv:2608.11655）

> 来源归档（ingest）

- **标题：** Motion-as-Prompt: Enhancing Motion Reasoning in Multimodal Large Language Models via Motion-Guided Cross-Frame Visual Prompting
- **缩写：** MaP
- **类型：** paper / video-reasoning / mllm / visual-prompting / training-free
- **arXiv：** <https://arxiv.org/abs/2608.11655>
- **项目页 / 代码：** <https://github.com/SunVictor23/MaP>（归档见 [`sources/repos/motion-as-prompt.md`](../repos/motion-as-prompt.md)）
- **入库日期：** 2026-08-19
- **一句话说明：** 恢复密集点轨迹，选运动信息帧，把累积轨迹画在视觉输入上；冻结 MLLM；CLEVRER / SSv2 运动推理涨点且不损非运动理解。

## 开源状态（步骤 2.5）

- **仓库核查（2026-08-19）：** `map_kit/` + benchmark runners + 多 GPU 脚本；**无 MaP 专用权重**（by design）。
- **外部依赖：** CoTracker3 checkpoint；本地 eval 可用 Qwen3-VL-2B；benchmark 数据来自 HF（MVBench、SSv2、TempCompass）。
- **结论：** **已开源、可运行**（训练无关框架）。

**对 wiki 的映射：** [`wiki/entities/paper-motion-as-prompt.md`](../../wiki/entities/paper-motion-as-prompt.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查
