# PI：FAST、MEM 与 RL Token 一手资料补核

- **类型：** 官方模型页 / 原始论文
- **核查日期：** 2026-10-05
- **FAST 项目页：** <https://www.pi.website/research/fast>
- **FAST tokenizer：** <https://huggingface.co/physical-intelligence/fast>
- **MEM 项目页：** <https://www.pi.website/research/memory>
- **MEM 论文：** <https://arxiv.org/abs/2603.03596>
- **RL Token 项目页：** <https://www.pi.website/research/rlt>
- **RL Token 论文：** <https://arxiv.org/abs/2604.23073>
- **原始官方索引：** [PI 技术文章索引](pi-website-technical-articles.md)

## 补核摘录

| 项目 | 从一手资料能归纳的机制 | 评测或工程读法 |
| --- | --- | --- |
| FAST | 对时间轴 DCT、系数量化和 BPE 压缩动作块；解码时恢复连续动作 | HF 页面提供可加载 tokenizer；跨本体使用仍需动作维度、归一化和采样率对齐 |
| MEM | 秒级视觉历史由视频编码器处理；分钟级语义事件由高层策略更新语言记忆，条件化低层 VLA | 论文基于 π₀.₆，报告最长约 15 分钟的厨房/烹饪任务；须对照无记忆、单尺度记忆消融 |
| RL Token | encoder–decoder 压缩 VLA 特征；适配后冻结 VLA 与表示，在线 actor/critic 修正动作块 | 论文用 π₀.₆，四项精密操作；报告成功率与每 10 分钟吞吐，不能只看峰值成功率 |

## 开源核查与日期口径

- **FAST：** 官方 HF tokenizer 有实现/模型入口；不等于后续 PI 通才策略的全套预训练资产。
- **MEM / RL Token：** 本次两篇官网页面返回 403；可阅读原始 arXiv 论文，**未确认完整官方训练/部署实现**。不要把“页面访问失败”写成“确认没有代码”。
- 公司路线沿用官网文章发布日期口径；RL Token 官网索引为 2026-03-19，而 arXiv 编号为 2026-04，代表不同发布事件，不以编号覆盖博客日期。

## 对 wiki 的映射

- **RLT 评测补核：** 原文 VI-A / VII：四任务主对比仅关键阶段，各 50 次、从部分完成状态重置；完整任务只补测螺丝/扎带。训练含人工奖励、纠正与策略切换；自主切换是可追加的训练方案，不等于四任务已完整自主运行。

- [FAST](../../wiki/entities/paper-rcl-2501-09747-fast-efficient-action-tokenization-for-vision-la.md)
- [MEM](../../wiki/entities/paper-pai-2603-03596-memmultiscaleembodiedmemory.md)
- [RL Token](../../wiki/entities/paper-rcl-2604-23073-rl-token-bootstrapping-online-rl-with-vision-lan.md)
