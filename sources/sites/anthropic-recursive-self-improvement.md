# When AI builds itself（Anthropic Institute）

> 来源归档

- **标题：** When AI builds itself
- **副标题：** Our progress toward recursive self-improvement, and its implications
- **类型：** site / institute essay（政策与内部证据综述，非评测报告）
- **来源：** The Anthropic Institute
- **作者：** Marina Favaro, Jack Clark（Santi Ruiz 编辑；可视化 Shan Carter 等）
- **链接：** https://www.anthropic.com/institute/recursive-self-improvement
- **入库日期：** 2026-08-28
- **一句话说明：** Anthropic 用公开基准与内部数据论证：AI 已在加速 AI 自身研发；**完全递归自改进（RSI）尚未发生、也非必然**，但可能比多数机构准备得更快；具身智能（机器人）被预期会 **跟随** 递归智能。
- **开源状态：** **不适用（论述文，无可运行实现）** — 无项目页代码；引用的内部生产率数字不可复现。公开锚点为 METR 时程、SWE-bench、CORE-Bench 等。
- **沉淀到 wiki：** [递归自改进](../../wiki/concepts/recursive-self-improvement.md)、交叉 [AI Auto-Research](../../wiki/concepts/ai-auto-research.md)

---

## 抓取说明

- 以 **2026-08-28** 对 Institute 页公开 HTML 正文抽取为准。
- 内部数字（代码占比、8×、52× 加速等）是 **公司自报**；文内已提示 LOC 高估真实生产率、主观 4× 可能偏高（METR 研究）。wiki 必须带这一层读法。
- 机器人相关只有「具身智能可能迅速跟随递归智能」一段，不要把全文当成机器人论文。

---

## 公开证据（文内）

| 指标 | 叙事 |
|------|------|
| METR 任务时程 | 约每 4 个月翻倍（早先约 7 个月）。Opus 3（2024-03）~4 分钟 → Sonnet 3.7 ~1.5 小时 → Opus 4.6 ~12 小时。Mythos Preview「至少」16 小时，已顶到 METR 现有任务上限。 |
| SWE-bench | 两年内从个位数饱和。 |
| CORE-Bench（复现已发表研究） | 2024 约 20% → 15 个月后饱和。 |
| 工程师产出 | 2026 Q2 典型工程师合并代码量约为 2024 的 **8×/天**（脚注：LOC 是数量不是质量）。 |
| 代码归因 | 2026-05 **>80%** 合入生产的行可归于 Claude（领导层口头 90%+ 含脚本；>80% 更保守）。Claude Code 2025-02 预览前为个位数。 |
| 主观 uplift | 2026-03 对 130 名研究员工：中位估计 Mythos 相对「无任何 AI」约 **4×**；作者认为真实值偏低。 |
| 实验优化微型环 | 固定正确性、尽量加速小训练代码：Opus 4（2025-05）~3× → Mythos（2026-04）~52×。熟练人类 4–8 小时约到 4×。绝对值依赖起点代码，宜看相对变化。 |
| 开放研究项目 | 弱模型监督强模型：两人约一周收回差距 **23%**；agents 累计 800 小时、约 $18k 算力收回 **97%**。未干净转移到生产规模；人类仍选题与打分。 |
| 下一步判断 | 129 个「人曾走弯路」的会话切片：Opus 4.5 优于人的下一步 **51%** → Mythos **64%**。对照集（人已经走对）上模型只约 20% 更好。 |

其余内部 vignette：一次升级搞崩数万训练任务，Claude 约 2 小时定位冷门 debug flag（人需 2–3 天）；2026-04 Claude 合入 800+ 修复、一类 API 错误降三个数量级；自动化 Claude review 回溯能在上线前抓住约 1/3 的 claude.ai 事故 bug。

---

## 三个未来情景（文内）

1. **趋势失速，但今日能力广泛扩散** — 指数实为 S 曲线；或能源/芯片/互连成约束。作者认为最不可能。即使能力冻结，Project Glasswing 已在关键系统找到上万高危漏洞，瓶颈从「找洞」变成「补丁跟得上」。
2. **实验室持续获得复合效率，人仍定方向** — 作者认为 **最可能正在进入**。Amdahl：人审代码已成为新瓶颈；想法爆炸超过组织追赶能力。
3. **系统具备完全 RSI，开始造后继者** — 进度由算力与算法效率决定；人对齐/验证虚拟实验室。误对齐可能在代际复合。即便如此，药监、选举、人际关系等瓶颈仍在。**具身智能（机器人）被预期会迅速跟随递归智能**，走类似「能力升、成本降」路径。

## 政策立场（文内）

若能有效放慢以换对齐与社会准备，作者认为那是好事；但单边暂停只换领跑者。可信暂停需要多国多实验室可验证停训——训练比导弹井更难探测。Institute 将推动验证基础设施的对话。

---

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| RSI 定义、三情景、具身跟随 | `wiki/concepts/recursive-self-improvement.md` |
| 人机共治 / 判断仍是瓶颈 | `wiki/concepts/ai-auto-research.md` |
| 真机/仿真 coding agent 闭环 | `wiki/queries/real-robot-policy-autoresearch-harness.md`、`wiki/methods/aspire.md`、`wiki/methods/enpire.md` |
| 规模与苦涩教训 | `wiki/concepts/embodied-scaling-laws.md`、`wiki/concepts/bitter-lesson.md` |

## 参考链接

- <https://www.anthropic.com/institute/recursive-self-improvement>
- METR 任务时程研究（文内引用）
- Claude Opus 4.7 System Card §2.3.5（文内指向 2026-03 调查方法）
