# Astronex-World 1.0（arXiv:2609.20034）

> 来源归档（paper）

- **标题：** Astronex-World 1.0: Real-Time Interactive World Model Foundation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.20034>
- **PDF：** <https://arxiv.org/pdf/2609.20034>
- **项目页：** <https://world.astronex.com.cn>
- **代码：** <https://github.com/Astronex-Robotics/Astronex-World>
- **权重/数据：** <https://huggingface.co/Astronex-Lab/Astronex-World>
- **入库日期：** 2026-09-19
- **一句话说明：** 5B 可控视频世界模型基座（Wan2.2-TI2V-5B 先验）：PRoPE 相机 + 64-D 动作/embodiment + 事件插入；双向/块因果两形态；2×L20 五阶段训练、1×L20 832×480@24fps 实时；WBench Full **70.0**、Navi **73.5**。

## 开源状态

- **已开源**（步骤 2.5，2026-09-19）：GitHub 推理/后训练代码 Apache-2.0；Hugging Face 权重；项目页公开技术报告与 demo。

## 核心摘录

1. **控制接口：** T2V/I2V 同权重；帧对齐相机内外参（PRoPE）、64-D 连续动作流、32 embodiment ID、指定帧 text event。
2. **因果流式：** 块因果注意力 + 跨块 KV cache；20 帧局部窗 + 4 持久 sink；8-step UniPC（默认）。
3. **训练五阶段：** 双向控制适配 → 块因果转换 → UniPC 轨迹蒸馏 → 混合域 SFT → 非对称 DMD/DMD2。
4. **Benchmark：** WBench Full 70.0 高于 13.6B LongCat-Video、14B Helios，接近 22B LTX-2.3；VBench 1.0 T2V/I2V 分项见项目页。
5. **机构：** Astronex Robotics、南京信息工程大学（NUIST）。

**对 wiki 的映射**

- [paper-astronex-world-1](../../wiki/entities/paper-astronex-world-1.md)
- [generative-world-models](../../wiki/methods/generative-world-models.md)
- [model-based-rl](../../wiki/methods/model-based-rl.md)
