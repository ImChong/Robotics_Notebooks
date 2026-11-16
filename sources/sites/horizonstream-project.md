# HorizonStream 官方项目页 — 一手核验

- **标题：** HorizonStream: Long-Horizon Attention for Streaming 3D Reconstruction
- **类型：** 作者项目主页
- **链接：** <https://3dagentworld.github.io/horizonstream/>
- **论文：** <https://arxiv.org/html/2605.23889v2>
- **会议：** NeurIPS 2026（项目页标示；arXiv v2 的 Comments 亦标示 Accepted）
- **代码：** <https://github.com/3DAgentWorld/HorizonStream>
- **模型：** <https://huggingface.co/NicolasCC/HorizonStream>
- **演示：** <https://huggingface.co/spaces/NicolasCC/HorizonStream_Demo>
- **核验日期：** 2026-11-10
- **开源状态：** **部分开源**。官方 GitHub 已公开推理、评估、可选闭环校正、交互演示和模型权重；官方 README 明示训练代码因公司政策暂不提供；代码仓库 LICENSE 为 CC BY-NC-SA 4.0，非一般商业可用的宽松许可。

## 主页和论文一致的关键点

- 输入为单目 RGB 视频流；输出为位姿、深度和稠密几何。
- Geometric Linear Attention（GLA）是跨时间的定长状态传播，Gated Window Attention（GWA）处理局部可靠匹配，Metric Readout Tokens（MRT）给尺度/位姿恢复提供读出。
- 论文报告 48 帧训练可处理 10,000+ 帧长序列；这是作者的实验结论，不是本归档的独立复现。
- loop closure 是可选步骤，必须把其结果与无 LC 纯在线结果分开对照。

## 相关归档

- [论文全文通读](../papers/horizonstream_arxiv_2605_23889.md)
- [官方代码运行入口](../repos/horizonstream-official.md)
- [Wiki 项目节点](../../wiki/entities/paper-horizonstream.md)
