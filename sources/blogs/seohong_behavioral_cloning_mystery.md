# Behavioral cloning mystery（Seohong Park）

> 来源归档（blog / 个人研究笔记）

- **标题：** Behavioral cloning mystery
- **类型：** blog
- **作者：** Seohong Park（UC Berkeley PhD，Sergey Levine；同时任职 Physical Intelligence）
- **原始链接：** <https://seohong.me/blog/behavioral-cloning-mystery/>
- **发表日期：** 2026-08
- **入库日期：** 2026-08-26
- **抓取方式：** 官方页直连 HTML（curl）；Jina Reader 超时
- **一句话说明：** 用脚本化、人类演示风格的仿真操作数据，在受控基准上复现真机 BC 的四条「民间观察」——过拟合有时更好、开环优于闭环、策略必须极大、特征缩放在无限数据下仍改变成功率——并主张根因是测试时分布偏移与非马尔可夫数据 vs 马尔可夫策略的表达力错配。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-08-26） |
|----|-------------------------|
| 本篇博客 | 个人站 `seohong.me`，无独立 `*.github.io` 研究项目页 |
| 复现基准 / 数据集 | **宣称将开源** — 作者写「计划今年 10 月发布官方基准」；入库日未见 GitHub / 数据下载链 |
| 代码 / 权重 | **未开源**（本篇实验配方未附仓库） |
| 可信度边界 | 作者自报仿真复现；明确声明现象本身不宣称新颖，目标是把民间观察做成可重复科学 |

## 核心摘录（归纳，非全文）

### 设定

- **动机：** 真机演示数据与 D4RL / OGBench 等仿真 RL 基准差异大；真机不可复现（光照、复位、温度），故用仿真脚本策略模仿「人类演示的统计性质」再做消融。
- **人类演示的关键性质：** 分布极窄、时间强相关（非马尔可夫）、平滑随机。
- **脚本策略：** 7-DoF 关节速度、50 Hz；分段 Hermite 样条 + 大量随机化（抓取角、接触点、速度、夹爪姿态）；偶发错误与恢复。
- **任务：** block stacking、汉诺塔、bowling 等；支持 **MJWarp**（GPU MuJoCo）→ 可「无限数据」流式生成、每条轨迹只训一次。
- **BC 配方：** 标准 flow matching、length-25 action chunk \(\pi(a_{t:t+24}\mid s_t)\)；状态基 MLP（非 VLA）。作者强调：**现象可在纯状态设定复现 → 根因在数据性质，不在视觉架构。**

### Mystery 1：过拟合往往不坏，有时更大数据集更差

- 10K episode 的 block pick-and-place：验证 flow loss 上升时，任务成功率仍上升并稳定。
- 同分布 10K vs 50K：10K 有时更好（4 seed × 4 独立采样，非偶然）。
- **假说 A：** 过拟合轨迹 ≈ 测试时沿最近邻片段执行，减小分布偏移；若「搜索」发生在已学表征上，仍可泛化（类比 LLM 模糊近邻）。
- **假说 B：** flow matching 损失与策略表现不对齐；验证 **action MSE** 在 flow loss 恶化时仍稳定。更相关的指标是 **策略诱导测试分布** 上的动作误差——通常拿不到标签。

### Mystery 2：开环优于纯闭环（即使无限数据）

- **开环：** \(\pi(a_{t:t+24}\mid s_t)\)，播完整 chunk 不重规划。
- **闭环：** \(\pi(a_t\mid s_t)\)，每步单动作。
- 无限数据下纯闭环完全失败（简单 block-single 甚至碰不到方块）。
- **原因 1：** 更短 horizon → 更频繁查询 → 复合误差（随机 flow 策略尤甚）；该任务甜区约 25 步。
- **原因 2（更关键）：** 环境可马尔可夫，**数据集因时间相关而非马尔可夫**；闭环学到「马尔可夫化」行为 → 测试分布偏移。
- **历史条件化对照：** \(\pi(a_t\mid s_{t-24:t})\) 与开环表达力对齐后，**历史条件化反而更差**（尽管 train loss/MSE 更好）。假说：因果混淆（抄上一动作）或更大输入空间更易偏移。

### Mystery 3：策略必须非常大

- 固定、非目标条件、37 维状态的 pick-and-place：`[512]*3` MLP 远不够。
- 至少 **`[4096]*8` residual MLP** 才学得好；8192 维进一步提升 → 约 **0.5B** 参数。
- 对照：D4RL / OGBench 通常 `[1024]*4` 已够。
- 作者无定论：BC 本身很难（暗示 VLA 即使感知完美也需要很大 **action expert**），或当前 flow BC 实现低效。自回归 tokenized 动作同样需要大模型。

### Mystery 4：无限数据下特征工程仍重要

- **Handcrafted 缩放** vs **各维标准化**：信息相同、train flow loss/MSE 几乎相同，成功率显著不同。
- 在标准化特征上把物体/夹爪 \(xyz\) 再乘 0.1 或 10：放大物体位置的策略更好。
- **解释：** 测试时分布偏移——关注物体的策略比关注关节角的策略更能泛化到未见状态。这与 [Bitter Lesson](../../wiki/concepts/bitter-lesson.md)「无限数据可放弃手工特征」在 BC 闭环里不自动成立。

### 收束观点

1. **根本解可能是回避问题：** 把数据 scale 到测试几乎都 in-distribution（解释 LLM 为何较少受此困扰、可用训练损失当代理）。VLA 若到 LLM 级规模，许多 mystery 或消失。
2. **表达力缺口：** 非马尔可夫数据 vs（chunked）马尔可夫策略。候选：全历史自回归策略；或分层——高层先输出消灭多模态的 plan，低层再更马尔可夫。
3. RL 侧同类现象留给后续博文/论文。

## 对 wiki 的映射

- [behavioral-cloning-mysteries](../../wiki/concepts/behavioral-cloning-mysteries.md) — 本篇升格概念页
- [behavior-cloning](../../wiki/methods/behavior-cloning.md) — 方法页补充真机数据现象
- [action-chunking](../../wiki/methods/action-chunking.md) — 开环 vs 闭环；对照 Revisiting Open-Loop
- [behavior-cloning-loss](../../wiki/formalizations/behavior-cloning-loss.md) — train loss ≠ 测试成功率
- [bitter-lesson](../../wiki/concepts/bitter-lesson.md) — Mystery 4 作为 BC 闭环下的反例边界

## 可信度与使用边界

- **仿真复现，非真机对照实验**；脚本策略「像人」只匹配统计性质，不等于真实遥操作。
- 定量图为作者自报；基准未发布前不可独立复现。
- Mystery 2 与 [Revisiting Open-Loop](../../wiki/entities/paper-revisiting-open-loop-action-chunking.md) 表面张力：后者主张加长观测上下文后闭环更优。本篇历史条件化失败，说明「加历史」≠「加对上下文」——对读时保留两条证据链。
