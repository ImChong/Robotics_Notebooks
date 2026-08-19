# τ₀-VLA 项目页（tau0-vla.github.io）

> 来源归档

- **标题：** τ₀-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation
- **类型：** site
- **URL：** <https://tau0-vla.github.io/>
- **论文 PDF：** <https://tau0-vla.github.io/tau0-vla.pdf>
- **组织：** 上海创智学院；智元机器人 Finch 团队；香港中文大学
- **入库日期：** 2026-08-19
- **一句话说明：** 官方项目页：分层 **子任务 + 低层 VLA**、**世界模型引导测试时计算**、四类 **13–25 步** 长程真机任务 rollout 与对照表；链到 arXiv、GitHub 与 Hugging Face。

## 页面要点（策展）

1. **问题动机：** 长程家务（奶茶、炒菜、打扫）从数秒扩展到数分钟；瓶颈从 **精细控制** 转向 **进度跟踪、后果预测与子任务规划**。
2. **分层接口：** 高层在 **子任务边界** 读指令、观测与 **执行记忆** 选下一子任务；低层以更高频执行所选子任务。
3. **TTC 环：** token 置信度决定快路径或额外算力；不确定时 **提出候选子任务 → 世界模型想象后果 → 价值模型打分 → 搜索/反思** 再提交。
4. **记忆修订：** 失败抓取或缺失物体可使记忆与物理不同步；可 **前进、回滚、重试** 修复滞后或过度乐观的进度记录（扰动记忆训练，无单独纠错集）。
5. **长程评测：** Clean Room（25 步）、Prepare Ingredients（14）、Stir Fry（22）、Make Milk Tea（13）；episode 最长 12 分钟。
6. **对照（项目页表，10 trials/task）：** τ₀-VLA 直出 **27.5%** avg；分层 Plan Once **45.0%**；π₀.₅ **22.5%**；GR00T N1.7 **2.5%**；LingBot-VLA **0%**。
7. **算力–精度：** TTC 在 OOD Book Organization next-subtask **74.0%**（Plan Once 50.0%，Best-of-N 57.5%）；低–中算力预算收益最大后渐饱和；置信路由实现 **选择性 TTC**。
8. **低层 generalist：** Qwen3.5 VLM + MoT action expert；**40 维** 统一接口；**40,115 h** 异构真机 + 多模态共训；ARX / Franka 跨本体 rollout 演示。

## 对 wiki 的映射

- [τ₀-VLA](../../wiki/entities/paper-tau0-vla.md) — 实体归纳页
- [sources/papers/tau0_vla_arxiv_2608_16885.md](../papers/tau0_vla_arxiv_2608_16885.md) — 论文摘录
- [sources/repos/sii_research_tau_0_vla.md](../repos/sii_research_tau_0_vla.md) — 官方实现
