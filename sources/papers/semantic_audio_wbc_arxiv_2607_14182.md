# Semantic Audio-driven Understanding for Dynamic Humanoid Whole Body Control（arXiv:2607.14182）

> 来源归档（ingest）

- **标题：** Semantic Audio-driven Understanding for Dynamic Humanoid Whole Body Control
- **类型：** paper / humanoid whole-body control / audio-driven / skill orchestration / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2607.14182>
- **arXiv HTML：** <https://arxiv.org/html/2607.14182v1>
- **PDF：** <https://arxiv.org/pdf/2607.14182>
- **项目页：** <https://lab-rococo-sapienza.github.io/semantic-WBC/>
- **代码：** <https://github.com/Lab-RoCoCo-Sapienza/semantic-WBC>（已开源）
- **机构：** 罗马大学 Sapienza、罗马国际大学（UNINT）
- **作者：** J. M. A. Marcelo、M. Brienza、E. Bugli、L. Comito、D. Nardi、D. D. Bloisi、V. Suriani
- **硬件：** Unitree G1（仿真 + 真机）
- **控制框架：** RoboJudo；技能库含 Walk / Stand 过渡策略 + BeyondMimic 模仿技能（ONNX 部署）
- **发表日期：** 2026-07-16
- **入库日期：** 2026-07-20
- **一句话说明：** 连续麦克风流经 **音乐/语音/跳过** 分层路由，音乐支路用 **音频指纹 + CLAP 嵌入** 做曲目与时序对齐并映射分段技能，语音支路 **ASR + 语义匹配** 触发模仿技能或 LLM 对话手势，经统一 TCP 接口在 **RoboJudo** 上调度 **BeyondMimic** 技能库，G1 仿真与真机验证 **84.8%** 块级检索准确率。

## 摘要级要点

- **动机：** 大量人形表演仍依赖时间码脚本；RL/模仿虽能学复杂动作，但缺乏 **何时执行何技能** 的语义编排。
- **音频路由：** AST（AudioSet 527 类）聚合 $p_{music}$/$p_{speech}$ + Silero VAD 帧占比 $v_{frac}$；互斥分到 Music / Speech / Skip。
- **音乐检索：** Wang 星座图音频指纹得曲目 ID、置信度、对齐票数与曲目内偏移 $\tau$；低置信回退 **CLAP** 余弦检索；$\tau$ 落入 timed rule 区间则选对应 whole-body 技能，否则曲目级默认映射。
- **语音分支：** gpt-4o-mini-transcribe 转写 → 技能库 top-1 语义匹配；未匹配则 LLM 回复 + TTS，并按语音时长触发等长手势策略。
- **执行层：** 新技能请求可经 **Stand 策略 priming** 再切换；MuJoCo G1 仿真与真机共用编排逻辑。
- **定量：** 574 个 5 s 音频块（含 0.5–2.0 s 起始偏移扰动）块级策略命中率 **84.8%**；M20（20 s 换曲）因过渡时间不足易失稳回退 Walk，M30（30 s）编排更一致。

## 核心摘录（面向 wiki 编译）

### 与离线编舞管线对比

| 维度 | 本文（在线音频编排） | 离线 music-to-dance 生成 |
|------|----------------------|--------------------------|
| 输出 | **调度已有 RL/模仿技能** | 生成 kinematic 序列 |
| 时序 | 指纹/嵌入 **在线对齐** | 离线编舞 |
| 交互 | **语音 + 音乐双模态** | 通常仅音乐 |
| 真机 | **G1 sim2real** | 多数停留在仿真/运动学 |

## 对 wiki 的映射

- 沉淀实体页：[语义音频驱动人形全身控制（arXiv:2607.14182）](../../wiki/entities/paper-semantic-audio-wbc-humanoid.md)
- 交叉补强：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md)、[Imitation Learning](../../wiki/methods/imitation-learning.md)

## 当前提炼状态

- [x] 路由、双分支检索、RoboJudo 执行与评测摘录
- [x] wiki 实体页与 `sources/repos/`、`sources/sites/` 互链
- [x] 官方 GitHub 已收录
