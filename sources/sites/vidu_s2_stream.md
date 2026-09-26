# vidu_s2_stream

> 来源归档（project site / product demo）

- **标题：** Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation
- **类型：** site
- **原始链接：** <https://vidu.com/vidu-stream>
- **入库日期：** 2026-09-20
- **机构：** 清华大学 · 生数科技（Shengshu Technology）
- **论文：** [arXiv:2609.11638](https://arxiv.org/abs/2609.11638)
- **API：** <https://platform.vidu.com/vidu-stream/doc>
- **GitHub（文档仓）：** <https://github.com/shengshu-ai/Vidu-S> — 归档 [`sources/repos/vidu-s.md`](../repos/vidu-s.md)
- **Hugging Face Papers：** <https://huggingface.co/papers/2609.11638>
- **联系：** vidus@shengshu.ai
- **一句话说明：** Vidu S2 官方产品页：S2-Avatar / S2-Editing 可玩 Demo + FAQ；**无模型权重下载**（集成走 API 或 GitHub 文档链）。

## 页面要点（2026-09-20 核查）

### Vidu S2-Avatar
- 实时语音交互、复杂动作控制（如跳舞）。
- 参考图引导：物体交互、换装、换背景。
- 720p+ 输出；支持离线数字人（预设脚本/动作或图+音频异步生成）。

### Vidu S2-Editing
- 实时风格 / 人物 / 背景 / 虚拟试衣编辑。
- 输入：摄像头、视频或图像；流式编辑可中途换参考图。

### 与 S1 差异（FAQ）
- Avatar：**540p→720p**；更强指令跟随（如跳舞）。
- Avatar vs Editing：前者 **语音→渲染全链路实时交互**；后者 **连续视频流 in/out 编辑**。

### 集成
- API 文档：`platform.vidu.com/vidu-stream/doc`（含 S2-Avatar / S2-Editing quick-start）。

## 开源结论（2026-09-26）

- **产品页：** 仍 **无** 权重/训练代码直链；**Demo + API** ✅。
- **官方 GitHub [Vidu-S](https://github.com/shengshu-ai/Vidu-S)：** 文档与概览仓（非本地推理包）。
- **综合：** **部分开源** — 见 [`sources/repos/vidu-s.md`](../repos/vidu-s.md)。

## 对 wiki 的映射

- [paper-vidu-s2](../../wiki/entities/paper-vidu-s2.md)
- [vidu_s2_arxiv_2609_11638.md](../papers/vidu_s2_arxiv_2609_11638.md)
