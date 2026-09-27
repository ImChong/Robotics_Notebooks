# VLA Bit-Flip Attacks（arXiv:2608.15475）

> 来源归档（paper）

- **标题：** Bit-Flip Attacks on Vision-Language-Action Models: Action-Decoding Architecture Shapes the Vulnerability
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.15475>
- **PDF：** <https://arxiv.org/pdf/2608.15475>
- **入库日期：** 2026-09-27
- **一句话说明：** 量化 VLA 的 INT8 权重易受 Rowhammer 类比特翻转；关键位集中在动作解码层，少量翻转即可让闭环任务失效。

## 开源状态

- **待核实**（步骤 2.5 核查，2026-09-27）

## 核心摘录

1. **公众号章节：** 安全防御（[多模空间第四篇盘点](../blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md)）
2. **机构：** 香港科技大学；阿德莱德大学；佐治亚理工；北卡罗莱纳大学教堂山；中国石油大学（华东）
3. **文内评测：** LIBERO、SimplerEnv；双臂实机
4. **导读要点：** 部署侧需把权重完整性纳入威胁模型；动作头架构决定脆弱性分布。

## 对 wiki 的映射

- [paper-vla-bit-flip-attacks-int8](../../wiki/entities/paper-vla-bit-flip-attacks-int8.md)
- [第四篇技术地图](../../wiki/overview/vla-weekly-trends-2026-08-10-part4-technology-map.md)
