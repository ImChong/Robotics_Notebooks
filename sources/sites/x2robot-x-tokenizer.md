# X-Tokenizer Project Page（自变量机器人）

> 来源归档

- **标题：** X-Tokenizer — A Multimodal Action Tokenizer for Vision-Language-Action Pretraining
- **类型：** site / project page
- **URL：** <https://x2robot.com/pages/x-tokenizer>（官网研究博客列表 <https://x2robot.com/blog> 标注 **2026.06.30**）
- **镜像项目页：** <https://x-square-robot.github.io/X-Tokenizer_projectPage/>（arXiv Comments 与 README 指向此页；内容与官网页一致）
- **论文：** [arXiv:2606.14752](https://arxiv.org/abs/2606.14752)（v1 2026-06-07，v2 2026-06-28；cs.CV）
- **代码：** <https://github.com/X-Square-Robot/X-Tokenizer>（Apache-2.0）
- **权重：** <https://huggingface.co/x-square-robot/X-Tokenizer>（`xtokenizer.pth`，Apache-2.0）
- **机构：** 自变量机器人（X Square Robot）；香港城市大学；清华大学
- **作者：** Miracle Kang、Lights Shi、Lucy Liang、Roy Gan、Dongxiu Liu、Pushi Zhang、Sylas Chen、Shawn Qin、Yinan Zheng、Jinliang Zheng、Hao Wang、Xianyuan Zhan、Hang Su
- **官方新闻：** 官网 /news 2026-07-02 条目「自变量发布跨模态具身动作分词器X-Tokenizer，多模态对齐能力提升13.5%，长程任务性能提升8.25%」（外链网易新闻）
- **入库日期：** 2026-10-09
- **一句话说明：** 官网页标语为 "A multimodal action tokenizer that doubles as a semantic interface between vision-language reasoning and continuous robot control"。页面为 Next.js 渲染，正文以内嵌 HTML 形式放在 RSC payload 中；本归档的数字取自该 payload 中的图表 `DATA` 与论文正文，二者一致。

## 开源核查（2026-10-09）

| 入口 | 状态 |
|------|------|
| Homepage | 已挂链：<https://x2robot.com/pages/x-tokenizer>；按钮链 Technical report / Code / Model weights / BibTeX |
| Code | **已开源（推理侧）**：[X-Square-Robot/X-Tokenizer](https://github.com/X-Square-Robot/X-Tokenizer)，Apache-2.0；`pip install -e .`，提供 encode/decode API、统计量 CLI、两个示例和两条合成 demo episode。首个 commit 2026-05-19，最近 commit 2026-06-18 |
| 训练代码 | **未发布**：仓库中没有预训练循环，也没有 MAM / 对比对齐 / 下一帧 VL 预测三个辅助头；Wall-OSS 下游共训脚本同样缺席 |
| Checkpoints | **已发布**：HF `x-square-robot/X-Tokenizer` 的 `xtokenizer.pth`，约 1.02 GB（`x-linked-size` 1,016,902,790 B），2026-06-18 建库；Wall-OSS + X-Tokenizer 下游策略权重在 HF 组织页未见 |
| Data | **未发布**：2.4M 轨迹预训练语料混合了自变量内部数据与公开数据集；真机约 3.5k 遥操作轨迹和约 480k grounding 样本未公开；训练用归一化统计量也不随包提供（README 写明需用户自算） |
| Paper | arXiv:2606.14752 |

## 页面内容要点

- **TL;DR**：把动作分词从「压缩–重建」改为「语义接口学习」；Encoder → SRQ → Decoder，顶层码 q₀ 表示意图，q₁₋₃ 承载运动学残差。
- **规模**：2.4M 轨迹、2.0B 动作帧、17 个机械臂族。
- **头条数字**：RoboTwin Hard 相对 Easy 只掉 −3.8（π0.5 掉 −5.9）；5 本体联合训练 Hard +10.4；真机 7 任务平均 77.4；相对 FAST 多模态 grounding +13.5%、长程 +8.25。
- **噪声动画**：24 帧 chunk、6 个子 chunk × 4 token；σ=0.006 时只有 1/6 的 q₀ 翻转。
- **图表 DATA（与论文 Fig.8 / 9 / 10 一致）**：
  - RoboTwin 2.0（Easy / Hard / Avg）：π0 65.9 / 58.4 / 62.15；π0.5 82.7 / 76.8 / 79.75；X-VLA 72.9 / 72.8 / 72.85；Wall-OSS+X-Tokenizer 84.7 / 80.9 / 82.80
  - 跨本体（Easy / Hard / Avg）：单本体 70.9 / 64.0 / 67.45；5 本体联合 77.9 / 74.4 / 76.15
  - 真机（Pick Up Cup, Push Towel, Distribute Blocks, Stack Bottle, Place Tape, Arrange Flowers, Turn On Light Switch, VQA, Avg）：
    - Wall-OSS 46 / 70 / 39 / 60 / 60 / 38.5 / 35 / 50.4 / 49.8
    - +FAST 61 / 90 / 58 / 80 / 100 / 57 / 65 / 75.7 / 73.0
    - +RVQ(no-aux) 58 / 90 / 47 / 80 / 90 / 51 / 68 / 79.4 / 69.1
    - +X-Tokenizer 73 / 100 / 50 / 80 / 100 / 68.5 / 70 / 85.9 / 77.4
- **页内口径差异**：结果卡副标题写 "Easy/Medium/Hard"，但图表只有 Easy / Hard；方法卡把 decoder 主任务写成 "masked action modeling reconstructs…"，论文中 MAM 作用于顶层离散码、重建由 decoder 的 ℓ1 等损失负责。以论文为准。

## 对 wiki 的映射

- 代码归档：[`sources/repos/x-tokenizer.md`](../repos/x-tokenizer.md)
- 沉淀 **[`wiki/entities/cn-os-x-tokenizer.md`](../../wiki/entities/cn-os-x-tokenizer.md)**
