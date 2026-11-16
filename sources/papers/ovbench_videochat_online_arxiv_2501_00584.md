# Online Video Understanding: OVBench and VideoChat-Online｜在线视频理解：OVBench 与 VideoChat-Online

- **类型：** CVPR 2025 paper，pp. 3328–3338
- **作者：** Huang Z, Li X, Li J, et al.
- **原文：** https://arxiv.org/abs/2501.00584
- **CVPR：** https://openaccess.thecvf.com/content/CVPR2025/html/Huang_Online_Video_Understanding_OVBench_and_VideoChat-Online_CVPR_2025_paper.html
- **项目：** https://videochat-online.github.io/
- **官方代码：** https://github.com/MCG-NJU/VideoChat-Online
- **官方权重：** https://huggingface.co/MCG-NJU/VideoChatOnline-4B
- **数据集：** https://huggingface.co/datasets/MCG-NJU/OVBench
- **在线训练数据：** https://huggingface.co/datasets/MCG-NJU/VideoChatOnline-IT
- **核对范围：** arXiv 论文主要方法、任务结构、训练及结果，仓库 README、流式评测脚本和现存训练入口；完整附录逐段复核尚未完成。
- **通读：** 未完成
- **入库日期：** 2026-11-16
- **wiki：** [论文实体](../../wiki/entities/paper-ovbench-videochat-online.md)

## 原文信息与 Wiki 映射

1. **OVBench：** 时间点嵌入视频流的问答，检查过去记忆、当前感知、未来推断，涵盖六类任务和十六子任务。
   - **对 wiki 的映射：** 强调按提问时刻约束可见帧，避免把离线整段视频输入当成在线能力。
2. **Pyramid Memory Bank（PMB）：** 在不同层次权衡空间分辨率与时间跨度，使用相似度检测及压缩控制记忆量。
   - **对 wiki 的映射：** 主流程图按视觉特征→分层记忆→在线回答。
3. **离线到在线训练：** 视频离线 SFT 后混合在线交错对话训练；提供不同评测协议。
   - **对 wiki 的映射：** 源码入口分别说明 SFT、streaming 和 sliding-window。
4. **结果与开源边界：** 官方 README 给出 VideoChatOnline-4B OVBench 54.9；代码当前公开 ViT 特征记忆实现，README 提示 KV-cache 特征版本尚待后续。
   - **对 wiki 的映射：** 不混淆在线/离线指标，不宣称 KV-cache 完整开源。

- [ ] 全篇含所有附录完成逐段核验后更新“通读”。
