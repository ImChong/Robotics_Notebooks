# semantic-WBC（Lab-RoCoCo-Sapienza 语义音频驱动人形编排）

- **标题：** Semantic Audio-driven Understanding for Dynamic Humanoid Whole Body Control — 官方实现
- **论文：** <https://arxiv.org/abs/2607.14182>
- **项目页：** <https://lab-rococo-sapienza.github.io/semantic-WBC/>
- **代码：** <https://github.com/Lab-RoCoCo-Sapienza/semantic-WBC>
- **类型：** repo / orchestration / audio-driven-humanoid
- **机构：** Sapienza University of Rome、UNINT
- **收录日期：** 2026-07-20
- **开源状态：** **已开源**（截至入库日 GitHub 可访问；项目页 BibTeX 互指）

## 一句话摘要

连续音频流 → **音乐指纹/CLAP** 或 **语音 ASR+语义匹配** → 统一技能 ID → **RoboJudo** 调度 **BeyondMimic ONNX** 技能与 Walk/Stand 过渡，在 Unitree G1 仿真与真机验证在线编舞。

## 为何值得保留

- **编排层开源样例：** 把「感知→技能选择」从离线 timecode 解放为 **embedding 检索**，与 manipulation 侧 VLM 技能路由形成对照。
- **与运球/足球栈互补：** 同 Lab-RoCoCo 组的 [learning-to-dribble 项目页](../sites/lab-rococo-learning-to-dribble.md) 走 RL 闭环，本仓走 **预训练技能库 + 语义调度**。

## 工程要点（公开 README / 论文对齐）

1. **5 s 固定块** 麦克风流；接受条件含指纹置信/票数阈值、VAD 主导度、技能冷却防振荡。
2. **音乐：** 星座图指纹索引曲目；$\tau$ 偏移映射 intro/verse/chorus 等 timed rules。
3. **语音：** 转写 → 技能库匹配；失败则 LLM+TTS 对话手势。
4. **执行：** RoboJudo TCP 接口；新技能可选 Stand priming；MuJoCo 仿真与 G1 硬件接口可切换。

## 关联资料

- 论文：[`sources/papers/semantic_audio_wbc_arxiv_2607_14182.md`](../papers/semantic_audio_wbc_arxiv_2607_14182.md)
- 项目页：[`sources/sites/lab-rococo-semantic-wbc.md`](../sites/lab-rococo-semantic-wbc.md)
- Wiki：[`wiki/entities/paper-semantic-audio-wbc-humanoid.md`](../../wiki/entities/paper-semantic-audio-wbc-humanoid.md)
- 技能训练参考：[BeyondMimic](../../wiki/methods/beyondmimic.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)
