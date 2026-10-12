# 框架与模型协同演化 论文来源归档

## 元数据

- **题目：** Co-Evolving Harnesses and Models: On-Policy Correction Helps Weaker Models Catch Up Where Imitation Fails
- **作者：** Zhou Yu, Bin Bi, Shiva Kumar Pentyala, Shubham Mehrotra, Sougata Chaudhuri, Shilpa Bhagavath, Zeyuan Chen, Ran Xu, Phil Mui, James Zhu, Sitaram Asur
- **单位：** Salesforce AI（arXiv HTML 署名页）
- **预印本：** https://arxiv.org/abs/2609.09134
- **HTML：** https://arxiv.org/html/2609.09134v1
- **版本：** v1，2026-09-08，cs.AI
- **代码与项目页：** arXiv 页与正文均未给出（2026-10-12 核查），按“无链接”处理。
- **周报线索：** [WaytoAGI 转载 DAIR.AI 周报](../blogs/waytoagi_feishu_top_ai_papers_2026-09-07-09-13.md)

## 论文要解决的问题

“harness”指模型外围的系统提示、工具、钩子、上下文管理与子智能体配置。企业场景中常先自动演化 harness，再对较小模型做轻量微调。论文发现：用强专家在演化后 harness 下的完整轨迹去模仿微调弱模型，七项企业任务全部下降（报告为 4–30 个百分点），原因是弱模型学到了自己执行不了的规划风格，与 harness 不再匹配。

## 方法与训练信息

- **任务（七项）：** 工资考勤审计（Attendance）、预算审批、股票告警、IoT 异常检测、浏览器自动化（Playwright）、网站管理（Webarena）、代码重构。
- **模型：** 弱模型为 Qwen3-Coder-30B-A3B-Instruct（总 30B、激活 3B）；Gemma-4-26B-A4B-it 用于复现；专家与 harness 反思/元智能体均为 gemini-3.1-pro-preview。
- **harness 演化：** GEPA 风格搜索，按失败驱动对提示、工具、钩子、上下文管理、子智能体配置提出编辑，仅当验证性能提升才保留；每任务三个种子，预算 20 美元，单次 rollout 上限 600 s。
- **微调：** LoRA-SFT，rank 16 或 64（取最佳），2 epoch，lr 1e-4，bf16，有效 batch 8，序列长度 49,152（截断超过 5% 时用 98,304），H100/H200 上不到一小时。
- **在策略纠正流水线：** (1) 收集弱模型在演化 harness 下自己的 rollout；(2) 自动化 MLE 智能体定位每个失败 rollout 的第一个失败轮；(3) 专家只重写该轮（一句策略加一次修正后的工具调用）；(4) 每轮采样 3 个候选，由质量评判选优；(5) 组成约 500 行数据（约 400 条纠正加 50 条自身通过样本）做 LoRA-SFT。

## 论文报告的结果（测试成功率 %，七任务平均）

| 设置 | 平均 |
|---|---|
| Qwen 基座，基线 harness | 29.2 |
| Qwen 基座，演化 harness | 78.0 |
| Gemini 专家，基线 harness | 84.4 |
| Gemini 专家，演化 harness | 93.6 |
| Qwen 模仿 SFT，演化 harness | 63.1（-14.9） |
| Qwen 模仿 SFT，基线 harness | 35.5（+6.3） |
| Qwen 纠正 SFT，演化 harness | 79.7（+1.7） |

逐任务（Attn / Budget / Stock / Anom / Playwright / Web / Refactor）：Qwen 演化 harness 81.7 / 87.6 / 76.7 / 89.4 / 97.3 / 43.3 / 70.0；模仿 SFT 51.9 / 77.0 / 62.6 / 85.2 / 81.3 / 23.0 / 60.9；纠正 SFT 81.3 / 87.0 / 78.9 / 91.1 / 98.5 / 48.9 / 72.2；Gemini 演化 97.0 / 93.2 / 94.4 / 100.0 / 100.0 / 85.6 / 85.0。
Gemma 复现仅 Webarena：演化 harness 46.7→55.6，模仿降到 41.1；Gemini 71.1→81.1。

说明：表中模仿 SFT 平均比演化 harness 基座低 14.9 个百分点，摘要所说“4–30 个百分点”是各任务降幅范围；两者口径不同，本文只引用表内数字。

## 复现与解释边界

- 论文无独立局限小节；文中承认纠正训练没有完全追平专家（可能受 LoRA 限制）；作者修改了环境（MCP 任务禁止直接改数据库、向优化器暴露工具错误），绝对值与先前工作（Yang et al., 2026）不可直接比；Gemma 复现只覆盖一个任务。
- 无公开代码，数据与 harness 配置未发布；专家模型为闭源预览版。
- 附录未单独核对。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-co-evolving-harness-models.md
- 方法入口：wiki/methods/dagger.md、wiki/methods/imitation-learning.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2609.09134>
- arXiv HTML：<https://arxiv.org/html/2609.09134v1>
