# OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation

- **类型：** paper
- **机构：** 小米具身智能研究团队（Xiaomi Embodied Intelligence），合作作者见论文
- **论文：** <https://arxiv.org/abs/2604.18486>；[全文 HTML](https://arxiv.org/html/2604.18486)，核对版本 arXiv:2604.18486v3（2026-05-08）
- **发表：** NeurIPS 2026（[作者主页发表列表](https://ldkong.com/index.html)）
- **项目主页：** <https://xiaomi-embodied-intelligence.github.io/OneVL/>
- **推理代码：** <https://github.com/xiaomi-research/onevl>
- **训练代码：** <https://github.com/GeorgeLuImmortal/OneVL_training>
- **核对范围：** arXiv v3 全文：引言、相关工作、方法 §3、训练 §4、实验 §5、局限 §6、结论 §7 及附录 §8 的方法、实现、评测协议和定性例子；对照项目主页及官方 README。
- **通读：** 已完成
- **入库日期：** 2026-11-09

## 原文关键内容

1. **目标：** 缩短显式 Chain-of-Thought（CoT）驾驶规划的串行生成时间，同时以语言和未来视觉解释对紧凑潜推理提供训练监督。
2. **骨干：** Qwen3-VL-4B-Instruct。视觉潜位置与语言潜位置插在轨迹答案前；视觉辅助解码器重构未来 t+0.5 s 和 t+1.0 s 的 Emu3.5 视觉离散 token，语言辅助解码器重构显式 CoT 文本，两个辅助解码器均用于训练。
3. **潜 token 计数的两个口径：** 论文 §3.2 定义 4 个视觉潜标记、2 个语言潜标记；由于复用现有词表而非新增 special token，文本编码后分别实现为 **35 个**和 **20 个模型 token**。官方推理入口的参数为 `--num_latent_vis 4 --num_latent 2`，并不与 35/20 矛盾。
4. **训练：** Stage 0 轨迹答案监督预热主模型；Stage 1 冻结主模型学习两个辅助解码器；Stage 2 联合微调。总目标为轨迹交叉熵加语言与视觉辅助损失；论文使用语言权重 1.0、视觉权重 0.1。
5. **推理：** 辅助解码器不进入主规划推理通路，潜标记经 prompt prefill 一次处理，随后自回归生成轨迹；如需解释，可另行调用辅助解码器，解释解码不应算作所报最快规划通路的必需步骤。

## 实验核对（作者报告，非独立复现）

| NAVSIM 实验 | PDM-score ↑ | 主规划延迟 s ↓ |
|---|---:|---:|
| 直接答案 AR Answer | 87.47 | 4.49 |
| 显式推理 AR CoT+Answer | 88.29 | 6.58 |
| OneVL | **88.84** | **4.46** |

- **语言解释质量：** OneVL 语言辅助解码平均分 76.13，显式 CoT 78.27；不是解释质量全面领先。
- **NAVSIM 消融：** 去视觉辅助解码器 87.97；去语言辅助解码器 88.53；去分阶段训练 **67.13**；完整 88.84。
- **ROADWork：** ADE 12.49 px、FDE 28.80 px、4.71 s（显式 CoT 为 13.18 px、29.98 px、10.74 s）。
- **四种基准口径不可直接混合：** NAVSIM 使用 PDM-score；ROADWork、Impromptu 和 APR1 使用不同数据单位及 ADE/FDE。以上速度是论文特定评测软硬件下结果。

## 局限及复现核验

- 三模型结构（主模型加两个辅助解码器）使训练显存及维护成本高，训练中采用分阶段策略、DeepSpeed；推理阶段只需主模型才能走最快规划通路。
- 语言解释分数低于显式 CoT；未来场景辅助监督不等同于已证明安全的现实驾驶世界模型。
- 论文报告的性能不能替代真车闭环评测或第三方复现。官方开放推理、训练代码和部分权重；数据/环境仍需按各基准单独配置。

## 对 wiki 的映射

- [OneVL 实体与流程](../../wiki/entities/paper-xiaomi-onevl.md)
- [OneVL 项目主页归档](../sites/xiaomi-onevl.md)
- [OneVL 官方源码归档](../repos/xiaomi-onevl.md)
