# The Robots Build Now, Too（Generalist AI）

> 来源归档（blog / Generalist AI 官方）

- **标题：** The Robots Build Now, Too
- **类型：** blog（官网栏目 Research）
- **作者 / 组织：** Generalist Team / Generalist AI
- **原始链接：** <https://generalistai.com/blog/the-robots-build-now-too>
- **发表日期：** 2025-09-24
- **入库日期：** 2026-10-09
- **抓取方式：** curl 读取官网 HTML（服务端渲染正文）后去标签核对
- **一句话说明：** 介绍内部评测任务 **one-shot assembly**：人先搭一个小乐高结构，机器人看一眼后端到端（像素 → **100 Hz** 动作）复制搭建，无任务专用工程、无额外指令；作者称据其所知是首个以端到端视觉运动控制拼装乐高的机器人。

## 开源 / 项目页核查

| 项 | 结论（截至 2026-10-09） |
|----|-------------------------|
| 项目页 | **无**；仅博文与视频 |
| 代码 / 权重 / 数据 | **未公开**；正文未给模型名称、成功率或评测次数 |
| 可信度边界 | 官方演示博文；「世界首个」为作者自述（as far as we know） |

## 核心摘录（归纳，非全文）

### 任务设定（自报）

- **One-shot assembly** 是其「最新内部评测任务之一」：人搭建小结构 → 机器人复制。
- 端到端：从像素到 100 Hz 动作；**no task-specific engineering, no custom instructions**——「看到你搭什么就复刻什么」。

### 为什么重要（作者列出的三点）

1. **视觉理解**：仅凭观察目标结构决定「搭什么」。
2. **灵巧性**：乐高拼装需 **亚毫米** 精度、再抓取、轻推与按压（在凸点对齐瞬间施力）；引用 SemiAnalysis 机器人自主分级观点，把此类力依赖、精细任务归入最高的 Level 4。
3. **序列推理**：每块砖需选对、定向、暂放并正确安装。

### 泛化边界（博文自述）

- 仅测试过 **4 种颜色、3 块 2×4 乐高砖** 组成的结构。
- 作者估算：若无色 3 砖组合为 1,560 种，则每块 4 色给出 4 × 4 × 4 × 1,560 = **99,840** 种可能组合（组合数为作者引用的估计，非实测覆盖数）。
- 称受 Research Preview 中「乐高抛掷」演示的读者建议启发。

### 文内外链

- 前序演示：<https://generalistai.com/blog/research-preview>
- 自主分级观点：SemiAnalysis「Robotics Levels of Autonomy」<https://semianalysis.com/2025/07/30/robotics-levels-of-autonomy/>
- 乐高组合数参考：<https://web.math.ku.dk/~eilers/LIFE5UK.pdf>

## 对 wiki 的映射

- [generalist-ai-robotics](../../wiki/entities/generalist-ai-robotics.md) — 公司页「one-shot 乐高拼装评测（2025-09）」小节与时间线
- [generalist-gen15-one-shot](../../wiki/entities/generalist-gen15-one-shot.md) — 2026-08 的 one-shot physical prompting；二者「one-shot」含义不同：本篇是 **看成品结构复制**，GEN-1.5 是 **以示范轨迹作上下文提示**

## 可信度与使用边界

- 无成功率、无试验次数、无失败分析；只有视频。
- 99,840 为组合空间上界估计，**不等于** 实测覆盖或成功率。
- 博文未说明所用模型版本（推测为 GEN-0 发布前的内部模型，未经官方确认）。
