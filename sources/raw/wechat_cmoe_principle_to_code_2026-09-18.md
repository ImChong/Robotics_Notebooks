# 万字逐模块解读王兴兴收藏的项目CMoE如何使得G1准确适应不同地形 |全文主线

> 原始抓取（微信公众号）

- **来源 URL：** https://mp.weixin.qq.com/s/l6cy5nodRTfORY8SKwXXDw
- **抓取日期：** 2026-09-18
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）

---

# 关注公众号后台发送【CMoE】获取飞书链接方便阅读

这篇论文比较有意思的地方有两点：一是借助 SwAV（ SwAV），将地形表征与expert gate 激活在prototype空间中对齐，使Gate能够根据地形组织不同 Expert 的分工；二是 VAE （ VAE）不重构当前输入，而是预测下一时刻的本体观测，从历史中提取具有动态预测能力的机器人状态表示。

（全文约 1.1 万字；结构化归纳见 [`wechat_cmoe_principle_to_code_2026-09-18.md`](../blogs/wechat_cmoe_principle_to_code_2026-09-18.md)。核心模块：lazy gating 问题 → β-VAE 下一帧预测 → 地形 AE → dense MoE（5 完整 actor + 共享 gate + 5 critic）→ SwAV/Sinkhorn 对比 → 单阶段 PPO；含 `cmoe_actor_critic.py` 157 维 `actor_input` 与 `gate_weights.detach()` 代码导读。）
