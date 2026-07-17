# ACT-2 Preview: Generalizing Reliability

> 来源归档（blog / Sunday Robotics 官方）

- **标题：** ACT-2 Preview: Generalizing Reliability
- **类型：** blog
- **作者：** Sunday Team
- **原始链接：** https://www.sunday.ai/blog/act-2-preview
- **发表日期：** 2026-07-17
- **入库日期：** 2026-07-17
- **抓取方式：** 官方博客页面直接抓取（WebFetch）
- **一句话说明：** Sunday Robotics 预览 **ACT-2** 家庭机器人基础模型：以 **规模化人类 sensorized 预训练** 缩小 in-domain / out-of-domain **泛化鸿沟**，使 **少量 in-house Memos 后训练** 可 **跨未见真实家庭迁移**；在声明 **Solve** 边界下报告 **叠衣 99.1%（785 次、零部署适配）** 与 **单示范 SFT 可泛化新折法**；平台为移动全身机器人 **Memo**。

## 核心摘录（归纳，非全文）

### 问题重框

- 机器人领域常见 **demo ≠ 可靠性**：同一成功率数字可能对应「单房间单衣物」或「跨家庭零适配」完全不同主张。
- Sunday 提出 **Solve** 标准：**Performance × Scope × Adaptation cost** 三者必须同时声明，才使进展可比较、可累积。
- ACT-2 相对 **ACT-1**（2025-11，长时程移动操作、未见家庭泛化、灵巧性）的增量是：**可靠性可随 in-house 迭代 hill-climb，且增益可泛化到野外**。

### ACT-2 Recipe（三支柱）

| 支柱 | 要点 |
|------|------|
| **规模化预训练** | 高质量、高多样性、**sensorized 人类数据**（自研采集硬件 + 策展 + 处理管线）；**非零机器人数据预训练** 叙事与 ACT-1 一脉 |
| **缩小泛化鸿沟** | 定义 **generalization gap = in-domain SR − out-of-domain SR**（同后训练流程）；预训练规模 ↑ → 鸿沟 ↓（表：0% 预训练 gap **82%** → 100% gap **0%**） |
| **高效后训练** | **单条示范 SFT** 可教新折法并泛化到 held-out 衣物；失败 case 可用同一机制快速 **post-training** 闭环（细节留待技术博文） |

**数据质量消融（同算力/同体积）：** 高质量子采样 100% flagship **99.1% SR** vs 均匀子采样 50% 仅 **64.1%**；validation loss 与 SR 强相关（作者拟合 $R^2\approx0.90$–$0.98$）。

### Laundry Folding Solve（评估边界）

| 维度 | 声明 |
|------|------|
| **Garments** | 常见可叠衣物 9 类（T 恤、长短袖、polo、无袖、衬衫、裤、打底裤、短裤；XXS–8XL；多材质/厚度/纹理）；**排除** 袜、内衣、胸罩等（配对/悬挂为主） |
| **Scenes** | **未见** 房间、床/台面、光照、机器人站位（床左/右/脚端） |
| **Initial configs** | 篮中、床上堆、地面；任意朝向与自然褶皱 |
| **Adaptation cost** | **零**：无 per-home 数据、无专家示范、无部署后训练；**固定 checkpoint** |

**性能（785 次自主尝试）：**

- **成功率 99.1% ±0.3%**（778 次完成折叠并入栈）
- 按衣物：短裤/厚长袖/薄长袖/polo/无袖 **100%**；衬衫最低 **94.7%**（n=19）
- 按环境：床上堆 **98.8%**、地面篮 **99.5%**、床侧篮 **100%**
- **折痕质量**：5 星制均值 **4.72/5**；**98.3%** 达 4–5 星；**73.8%** 满分
- **速度**：成功折 median **2:13**，mean **2:19**（含自主重试与恢复）

**涌现行为（定性视频）：** 地面捡衣、扰动下重规划、儿童互动、明暗光照、**全身移动** 扩展工作空间（8XL 衬衫、婴儿服、大毛巾 vs 固定桌面臂）。

### 其他在训能力（未达 Solve 标准）

吸尘、玩具整理、拉拉链、裤子翻面、咖啡准备等——博客仅演示，**尚未按 Solve 框架完整评测**。

### 公司与部署

- **全栈：** 自研机器人 **Memo**、模型、机队基础设施、数据运营端到端闭环。
- **2026 秋** 通过 Beta Program 向家庭部署 Memo。
- 前身 **ACT-1** 博客/技术报告见脚注；**Series B** 等融资见同站其他博文。

## 对 wiki 的映射

- [sunday-robotics-act2](../../wiki/entities/sunday-robotics-act2.md)（ACT-2 系统实体 + Recipe / Solve 评测 Mermaid）
- [robotics-solve-standard](../../wiki/concepts/robotics-solve-standard.md)（Solve 评测框架概念页）
- [sunday-robotics](../../sources/sites/sunday-robotics.md)（项目站与开源核查）
- 交叉：[Manipulation](../../wiki/tasks/manipulation.md)、[VLA](../../wiki/methods/vla.md)、[TidyBot2](../../wiki/entities/tidybot2.md)、[Curr-0](../../wiki/entities/current-robotics-curr0.md)、[Humanoid Hardware 101 · 末端](../../wiki/overview/humanoid-hardware-101-sensing-end-effectors.md)

## 可信度与使用边界

- **公司官方预览博文**，非同行评审；定量结果依赖 **自研 grading rubric** 与 **内部 annotator**，需独立第三方复现。
- **评估 homes / 衣物未用于 task-specific post-training** 为作者声明（脚注 3），外界难以审计数据隔离。
- **单示范 SFT** 与 **generalization gap** 曲线缺少完整方法学与 checkpoint 公开细节（承诺后续技术博文）。
- **截至 2026-07-17**，[sunday.ai](https://www.sunday.ai/) **未列出** GitHub / Hugging Face / 模型权重（见 `sources/sites/sunday-robotics.md`）。

## Citation

```bibtex
@article{sunday2026act2preview,
  author  = {Sunday Robotics},
  title   = {ACT-2 Preview: Generalizing Reliability},
  journal = {Sunday Robotics Blog},
  year    = {2026},
  month   = {jul},
  url     = {https://www.sunday.ai/blog/act-2-preview}
}
```
