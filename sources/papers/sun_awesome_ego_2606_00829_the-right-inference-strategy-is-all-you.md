# The Right Inference Strategy Is All You Need: Nearly Training-Free Domain-Wise Inference for EgoCross Challenge

> 来源归档（深读 · arXiv:2606.00829 · EgoCross source-limited track）

- **标题：** The Right Inference Strategy Is All You Need: Nearly Training-Free Domain-Wise Inference for EgoCross Challenge
- **作者：** Leyi Wu, Yifan Zhao, Jinjie Zhang, Yinchuan Li, Ying-Cong Chen（HKUST(GZ) / Knowin；队名 WFJ-KnowinEnvision）
- **类型：** paper / egocentric-vision / vlm / benchmark
- **arXiv：** 2606.00829 · <https://arxiv.org/abs/2606.00829>
- **代码：** <https://github.com/YUEVII/Egocross-Challenge>
- **入库日期：** 2026-08-10
- **深读更新：** 2026-09-21
- **一句话说明：** 固定 Qwen3-VL-4B + 仅 20 条官方训练样本下，按 **四域分别设计推理接口**（帧采样、提示、logprob 验真、路由），overall **66.98%**，证明瓶颈常在 **迁移接口** 而非基座容量。

## 核心摘录（面向 wiki 编译）

### 1) 问题设定（EgoCross source-limited）

- **要点：** 测试域含手术、工业装配、极限运动、动物佩戴相机；与日常 egocentric 差异大。赛道 **锁定基座 Qwen3-VL-4B**，任务数据 **仅 20 样本** → 核心是如何 **暴露** 已有 VLM 知识，而非换更大模型。
- **对 wiki 的映射：** [`wiki/entities/paper-sa-2606-00829-the-right-inference-strategy-is-all-you-need-nea.md`](../../wiki/entities/paper-sa-2606-00829-the-right-inference-strategy-is-all-you-need-nea.md)

### 2) Domain-wise 推理程序

- **Animal：** 冻结基座；按题型分支（识别/交互/时序定位）；视频接口 1.0 FPS；确定性解码。
- **Surgery：** 冻结基座；CholecTrack20 + EgoSurgery 分题型：直接 MCQ、yes/no 验真、坐标回归、最早起始检测。
- **Industry：** ENIGMA 九路 **确定性专家路由**（2×SFT 时序定位 + 多路 base 计数/空间/下一交互）；0.5 FPS + 帧时间前缀；≤10 有效帧。
- **XSports：** 官方 **2 epoch SFT** checkpoint；8 帧均匀采样；默认 **option-guided yes/no logprob**；特殊动作走 **pairwise A/B margin** 六对聚合。
- **对 wiki 的映射：** 同上

### 3) 训练用量与最终精度

- **要点：** Surgery + Animal **零参数更新**；XSports + Industry 仅用官方 **20 样本、2 epoch SFT**。Table 1：Animal **77.05%**，XSports **63.41%**，Industry **64.49%**，Surgery **65.72%**，Overall **66.98%**。
- **对 wiki 的映射：** 同上

## 开源边界（步骤 2.5）

| 状态 | 说明 |
|------|------|
| **已开源** | [Egocross-Challenge](https://github.com/YUEVII/Egocross-Challenge) |
| **权重** | 基座与官方 SFT ckpt 以赛方发布为准 |

## 对 wiki 的映射

- [paper-sa-2606-00829-the-right-inference-strategy-is-all-you-need-nea.md](../../wiki/entities/paper-sa-2606-00829-the-right-inference-strategy-is-all-you-need-nea.md)
- [awesome-egocentric-vision.md](../../wiki/entities/awesome-egocentric-vision.md)
- [egocross-challenge.md](../repos/egocross-challenge.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2606.00829>
- 代码：<https://github.com/YUEVII/Egocross-Challenge>
- Benchmark：EgoCross（AAAI 2026）
