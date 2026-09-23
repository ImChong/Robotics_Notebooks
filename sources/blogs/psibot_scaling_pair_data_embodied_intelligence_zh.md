# Scaling Pair Data for Embodied Intelligence（PsiBot 技术博客 · 中文）

> 来源归档（blog / 公司技术长文）

- **标题：** Scaling Pair Data for Embodied Intelligence / 为具身智能扩展 Pair Data
- **类型：** blog / technical report / product demo（PsiBot 官方站，非 arXiv）
- **作者 / 组织：** 灵巧智能（PsiBot / Psi Robotics）
- **原始链接：** <https://www.psibot.ai/scaling-pair-data-for-embodied-intelligence-zh/>
- **英文版：** 同路径 `-zh` 后缀为中文页（站点另有英文生态）
- **入库日期：** 2026-09-23
- **抓取方式：** WebFetch 正文 + 项目页演示描述
- **一句话说明：** PsiBot 发布 **Psi-R2.5** 并系统阐述 **强 pair data** 如何把人–机 dynamic 对齐到同一 domain；从 **Psi-W0** 世界模型 RL 蒸馏出端到端人类→机器人转换器，并支持 **逆向** 从真机数据生成配对人手视频；配套 10 万小时质量复盘、50 任务真机评测与 ICL 演示。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-09-23） |
|----|-------------------------|
| 博客 / 演示入口 | <https://www.psibot.ai/scaling-pair-data-for-embodied-intelligence-zh/> |
| 公司首页 | <https://www.psibot.ai/>（Psi Sim / Psi Data / PsiR0 / PsiR0.5 / 将推 PsiR1 等） |
| GitHub / Hugging Face | **确认未开源** — 博客与公司首页未见 R2.5、pair 转换器或 W0 训练/推理仓库 |
| 权重 / 数据 | **未公开**（宣称自采 **10 万小时** 高质量人类数据 + 强 pair 管线） |
| 联系 | market@psirobot.ai |
| 关联开源（不同产品线） | PKU–PsiBot [EgoSteer](https://github.com/egosteer/EgoSteer)（arXiv:2607.09701）已开源，**非** 本文 R2.5 栈 |

## 核心摘录（归纳，非全文）

### Psi-R2.5 架构

| 层 | 骨干 | 输入 / 输出 |
|----|------|-------------|
| 上层 | **QwenVL3.5-4B** + 自采预训练 | 长 instruction → subtask 分解；含 memory、soft prompt 等 meta context + RL **value**；与当前观测一起喂下层 |
| 下层 | **Wan2.2-IT2V-5B** + 自采预训练 | 输出机器人 **操作轨迹**（视频–动作 WAM 风格） |

相对 **Psi-R2**：核心升级是 **数据质量提升 + 数据属性压缩**（非继续堆量）；在 **10 万小时** 规模做全量复盘，保留高质量子集；仿真测评嵌入预训练，真机 **50 个复杂多任务** 评测集 + 随机化初始状态。

### 强 pair vs 弱 pair

| 类型 | 定义 | 局限 |
|------|------|------|
| **弱 pair** | 相同 **任务语义**（如都拿同一瓶可乐）的人–机片段 | 场景/时序/动力学不对齐，难直接学 cross-embodiment |
| **强 pair** | 除本体外场景基本一致、**逐帧时序对齐**、action **可在机器人 replay** | 以往极难采集；R2/W0 管线 + 蒸馏转换器后可 scale |

### 数据哲学

- 具身预训练：**架构迭代快（2–3 天一轮），数据才是上限**；混训 raw 数据会隐式学「人类 vs 机器人」分类器。
- **任务多样性 > 单任务时长**：10000 h × 100 任务 × 1 h/任务 的信息量可能 ≈ 100 h × 100 任务 × 1 h/任务；压缩同小时数内冗余，扩任务数 + 精细原子动作标注。
- **人类数据质量门控**：① replay 轨迹能否在真机完成任务；② 后训练能否泛化——「训不出 policy 的人手数据不应进预训练」。
- 低质量人类数据在 fixed eval 下 **有害**（噪声 > 信息）。

### Embodiment gap 拆解

1. **Visual Embodiment Gap** — 人手 vs 机器人像素、相机、环境差异。
2. **Dynamic Embodiment Gap** — 手姿估计误差、运动学差异、摩擦等物理参数差。

业界路线：real2sim + sim2real；或 inpainting 换机械手（遮挡/穿模难处理）。PsiBot：**Psi-W0** 用 **~10 万小时** 人类数据预训练的世界模型，在 WM 内 RL 将人手轨迹优化为可执行机器人轨迹（替代传统仿真器，绕开 real2sim scale 与 sim2real gap）。

### 强 pair 生产管线（演进）

1. **R2 + W0**：WM 内 RL → 带 action、可 replay 的机器人轨迹（流程重：Policy↔WM rollout + RL）。
2. **蒸馏**：收集 W0 产出的强 pair → 训练 **端到端人类→机器人转换模型**（带 action 的 video editing 任务）。
3. **逆向（关键转折）**：从 **已有机器人数据** 出发，用 **逆 Psi-W0** 生成匹配人手数据 → 大规模强 pair；再训转换模型做人→机对齐。
4. **泛化**：用户 **手机随手录制** ego 人手视频 → 管线转为机器人数据（博客展示日常场景重建与物理一致性）。

### 后训练与 ICL

- **HIL + RL 后训练框架**（灵巧手）：基础模型 + 极少量数据微调；手机盒装配 from scratch IL 低成功率 → 数轮迭代 **~99%**，**1–2 工作日**；客户现场 corner case 回流。
- **In-Context Learning**：人类示教经 pair 模型转为机器人 context/prompt，**不更新权重** 完成新任务 zero-shot（R2.5 四段 ICL 演示视频）。

## 对 wiki 的映射

- 新建：[psibot-r25](../../wiki/entities/psibot-r25.md)、[strong-pair-data](../../wiki/concepts/strong-pair-data.md)
- 交叉：[egoscale](../../wiki/methods/egoscale.md)、[paper-egosteer](../../wiki/entities/paper-egosteer.md)、[world-action-models](../../wiki/concepts/world-action-models.md)、[robot-in-context-learning](../../wiki/concepts/robot-in-context-learning.md)

## 当前提炼状态

- [x] 博客 + 公司页核查
- [x] 开源状态：R2.5 / pair 转换器 **未开源**
- [ ] 若官方发布代码再补 `sources/repos/`
