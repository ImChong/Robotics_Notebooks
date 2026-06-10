# OMG 项目页（tsinghua-mars-lab.github.io/OMG）

> 来源归档（ingest 配套站点）

- **URL：** <https://tsinghua-mars-lab.github.io/OMG/>
- **标题：** OMG: Omni-Modal Motion Generation for Generalist Humanoid Control
- **机构：** Tsinghua University（清华大学 MARS Lab）
- **配套仓库：** <https://github.com/tsinghua-mars-lab/OMG>
- **入库日期：** 2026-06-10
- **论文：** 项目页 BibTeX 标注 **TBD**（截至入库日未见 arXiv 条目）
- **一句话说明：** 官方落地页：提出 **generator–tracker 分层** 的通用运动生成框架，OMG-DiT 将语言 / 音频 / 人体参考 / 运动历史等多模态意图实时转为 Unitree G1 可跟踪全身轨迹；配套 **OMG-Data**（约 1174.66 h）与真机多模态切换演示。

## 页面要点（2026-06 快照）

### 问题与定位

- 人形全身控制进展快，但 **RL 窄技能 + 重奖励工程** 与 **motion tracking 推理期强依赖参考** 并存，缺少把 **高层多模态意图 → 未来机器人运动** 的 **motion generator** 层。
- OMG 用 **generator–tracker 分层**：**OMG-DiT** 从语言、音频、人体参考、运动历史及其组合预测未来轨迹；**预训练 motion tracker** 将参考转为 G1 上物理可执行运动。

### 四大贡献（页面归纳）

1. **OMG 框架** — 统一多模态人类意图信号，经 generator–tracker 映射到可执行轨迹。
2. **OMG-Data** — 经重定向、过滤、标注与对齐到 G1 运动空间的大规模 omni-modal 人形语料。
3. **OMG-DiT** — 扩散 Transformer 骨干，语言 / 音频 / 人体参考等可组合条件。
4. **Foundation model 行为** — SOTA omni-modal 控制、模型 scaling、样本高效适配、控制信号零样本组合。

### OMG-Data 规模（页面数字）

| 子集 | 时长 |
|------|------|
| 总处理数据 | **1174.66 h** |
| 文本标注运动 | **1166.6 h** |
| 人体参考运动 | **958.77 h** |
| 音频条件运动 | **191.6 h** |

管线：聚合异构图形与人形数据集 → 校验分段 → 重定向到 G1 → **simulation-in-the-loop** 筛除物理无效轨迹。

### OMG-DiT 条件机制（页面归纳）

- **运动先验与条件模态解耦**：共享去噪 Transformer 建模可行 G1 运动；各模态经专用 encoder + cross-attention / FiLM / classifier-free guidance 注入。
- **语言**：运动历史 + 语言指令 → 全局 context token。
- **音乐 / 音频**：帧对齐音频调制节奏、 timing 与风格。
- **人体参考**：逐帧 motion guidance，充当 **neural retargeter** 角色。
- **组合控制**：单镜头演示可在 text / human ref / audio / text+audio 间 **实时切换** 而无需重置行为。

### 演示分组

- Text-conditioned whole-body motion（含 stylized 动作与 locomotion）
- Audio-conditioned motion（Liebestraume 等）
- Reference-guided control（人体参考 → 机器人执行）
- Real-time modality switching（一镜到底多模态切换）

## 对 wiki 的映射

- 实体页：[wiki/entities/paper-omg-omni-modal-humanoid-control.md](../../wiki/entities/paper-omg-omni-modal-humanoid-control.md)
- 仓库归档：[sources/repos/omg-tsinghua-mars-lab.md](../repos/omg-tsinghua-mars-lab.md)
- 交叉：[扩散运动生成](../../wiki/methods/diffusion-motion-generation.md)、[Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)、[HoloMotion](../../wiki/entities/holomotion.md)（官方 tracker 依赖）
