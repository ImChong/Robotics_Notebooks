# GEN-1.5: Embodied Foundation Models are One-Shot Learners（Generalist AI）

> 来源归档（blog / Generalist AI 官方）

- **标题：** GEN-1.5: Embodied Foundation Models are One-Shot Learners
- **类型：** blog
- **作者 / 组织：** Generalist Team / Generalist AI
- **原始链接：** <https://generalistai.com/blog/gen-1.5>
- **发表日期：** 2026-08-19（博客 Citation 写 Aug 2026）
- **入库日期：** 2026-08-20
- **抓取方式：** WebFetch；官方页 HTML 交叉核对
- **一句话说明：** Generalist **GEN-1.5** 宣称在 **8 个月以上** 持续预训练后，**无需显式 ICL 训练** 即涌现 **one-shot / few-shot 物理技能学习**：3–12 秒单次示范作 **physical prompt** 平均 **59%（±10%）** 成功率；**10 步梯度** + 5 分钟数据（~50 示范）达 **83%（±9%）**；并报告组合泛化、零样本 sim2real 提示、人→机示范迁移与即兴工具使用。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-08-20） |
|----|-------------------------|
| 本篇博客 / 项目页 | **无**独立研究项目页；入口为 <https://generalistai.com/> 与公司博客 |
| 代码 / 权重 | **确认未开源**（公司站与 GEN-1.5 正文未见 GitHub、Hugging Face 等公开训练/推理入口） |
| 数据集 | **未公开**（宣称 in-house 物理交互数据引擎；**预训练不含仿真数据**） |
| 可信度边界 | 产业官方博客，非 peer-reviewed；定量多为自报演示与内部分析 |

## 核心摘录（归纳，非全文）

### 主张与定位

- **GEN-1.5** 为 Generalist 最新具身基础模型，前序 [GEN-1](https://generalistai.com/blog/gen-1)、[GEN-0](https://generalistai.com/blog/gen-0)。
- 类比 GPT-3 在语言侧的 one/few-shot：**物理技能** 亦可在 **无梯度更新** 下从单次示范学习（「physical prompting」）。
- 10 项短程灵巧任务：one-shot **59%（±10%）**；10 步微调 **83%（±9%）**——作者称这是其已知首个在 **广泛任务** 上展示该能力的模型（任务仍简单、成功率中等）。

### 模型接口（博客披露）

| 维度 | 要点 |
|------|------|
| 模态 | 视频（**30 秒记忆**）、其他传感、语言、本体感觉 |
| 输出 | **100 Hz** 动作轨迹 |
| 上下文 | **30 秒** 窗口；示范占 3–12 秒，其余为滚动观测 |
| Physical prompt | **sensorimotor 序列**（传感 + 动作轨迹）；可来自人手示范（手持夹具）或机器人 rollout |
| 预训练 | 与 GEN-1 并行启动，**持续训练 8+ 个月**；随机连续片段采样，**无**为 ICL 定制的架构 / MAML / 辅助目标 |

### 能力簇

1. **One-shot in-context（physical prompting）** — 插入单次示范即执行，无训练。
2. **组合泛化** — 两个独立任务示范放入上下文可 **链成长程行为**，自行补中间重定位/纠错。
3. **零样本 sim2real 提示** — 预训练 **无仿真数据**，但可用 **纯仿真 rollout** 作 physical prompt 驱动真机；泛化到新手、新位姿/尺寸。
4. **人→机 in-context** — 部分任务人用手在相机视野示范，机器人可复现。
5. **极少步微调** — **1–10 梯度步**、**1–5 分钟**数据（~10–50 示范）；10 步权重变化 **<0.15%**；1 步 + 1 分钟数据 held-out **66.5%**。
6. **物理即兴** — 未见工具（刷子→香蕉、簸箕）、障碍清除、双手拧盖、自发分类等；微调步数越少越依赖预训练先验。

### 与 GPT-3 类比（作者自述）

- 语言 one-shot ~45%、few-shot ~65%（Brown et al. 2020）；物理 one-shot 59%、few-shot 83%（不同任务族，**不可直接数值对比**）。

### 涌现解释（假说，非定论）

- 物理观测/动作分布可能有 **burstiness / Zipf** 结构（类比语言 ICL，Chan et al. 2022）。
- 物理工作含重复周期，模型可能学会 **检测并延伸模式**（类比 pattern machines，Mirchandani et al. 2023）。

### 产品/使命叙事

- 超过某预训练阈值后，适应成本「可忽略」——几秒示范或 1 步微调更像 **提醒模型已知之事**。
- 若示教即可用，改变 **上手时间（秒级）** 与 **使用者范围（非专家）**。

## 对 wiki 的映射

- [generalist-gen15-one-shot](../../wiki/entities/generalist-gen15-one-shot.md) — 本篇升格实体页
- [paper-host-one-shot-human-video](../../wiki/entities/paper-host-one-shot-human-video.md) — 开源单视频对照（2026-09-04）
- [generalist-ai-robotics](../../wiki/entities/generalist-ai-robotics.md) — 公司入口页更新 GEN 系列脉络
- [generalist-gen1-thousand-hands](../../wiki/entities/generalist-gen1-thousand-hands.md) — 同公司 GEN-1 姊妹能力轴
- [foundation-policy](../../wiki/concepts/foundation-policy.md) — 商业通才策略 one-shot 对照
- [embodied-scaling-laws](../../wiki/concepts/embodied-scaling-laws.md) — 预训练时长与适应成本曲线
- [manipulation](../../wiki/tasks/manipulation.md) — 灵巧操作与示范学习
- [hub-cross-embodiment](../../wiki/overview/hub-cross-embodiment.md) — 人→机 / sim→real 提示迁移轴

## 可信度与使用边界

- **官方营销 + 技术叙事博客**；10 任务、短视频演示，无第三方基准表。
- **无公开权重/数据配方**；「首次」「涌现」等表述为作者立场。
- sim2real 此处指 **in-context 用仿真示范提示真机**，非传统「仿真训练策略直接部署」。
- 与开源 one-shot 文献（RoboTTT、Instant Policy、MINT-Zero 等）对照时须区分 **闭源规模** 与 **可复现协议**。

## Citation

```bibtex
@article{generalist2026gen15,
  author = {Generalist Team},
  title = {GEN-1.5: Embodied Foundation Models are One-Shot Learners},
  journal = {Generalist AI Blog},
  year = {2026},
  note = {https://generalistai.com/blog/gen-1.5}
}
```
