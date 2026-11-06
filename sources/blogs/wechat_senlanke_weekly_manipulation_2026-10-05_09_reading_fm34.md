# senlanke 周报 10-5–10-9 Manipulation 34 篇 通读

> 来源归档（原文通读 · 支撑 `wiki/overview/frontier-manipulation-2026-10-05-10-09.md`）

- **标题：** 【10.5-10.9前沿论文动态】Manipulation34篇
- **作者：** senlanke / 具身运控lab（微信公众号），2026-10-12 发布
- **类型：** blog / weekly-digest
- **原文：** <https://mp.weixin.qq.com/s/oPAUE5EvKeOfGU-Vjsynrw>（2026-11 用 `scripts/fetch_wechat_article.py` 取得全文，约 1.3 万字符）
- **一句话说明：** 按 10 个分栏列出 34 篇 Manipulation 论文，每条含英文标题、中文标题、机构、论文链接、「解决的问题」「创新点」。
- **核对范围：** 通读公众号全文 10 个分栏、34 个条目的「解决的问题 / 创新点」及文末导航；未逐篇打开各 arXiv 原文（技术事实以各详情节点引用的 arXiv 原文为准），机构信息为周报所述、多数标注「待核实」。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **分栏与篇数：** VLA 4；WAM / World Model 7；在线/离线 RL 与后训练 3；模仿学习与生成式动作策略 3；双臂 2；灵巧 2；柔性/可变形 2；人形上半身 4；长程任务、推理与规划 3；数据集、预训练、跨本体与硬件 4；合计 34。
- **VLA 栏：** ViDAL 用视觉动力学约束动作潜空间；EgoLAP 经语言-动作推理利用第一视角人类数据；VLA-ZO 冻结视觉语言部分、仅对动作侧做零阶优化，适应速度约为基础零阶方法的 25.59–32.54 倍，LIBERO 相机视角变化平均成功率 48.27% 到最高 63.58%；PLaW-VLA 引入预测式潜在世界建模（CoRL 2026 接收）。
- **WAM 栏：** OpenWAM 基于 Wan2.2-5B、超 1 万小时机器人视频因果预训练，LIBERO-Long 上 Video-to-Action 由 68.4% 到 97.8%；LAC-WM 用共享潜在动作空间统一跨本体动作，灵巧任务最高 +46.7%；MiniWAM 的 PRISM 编码使未来特征 Token 减少 65 倍、训练最高加速 8 倍（0.25B 参数）；RealtimeWAM、WAM-Cache 面向推理效率；Long-WAM 扩上下文；UNITAS 走原生三维路线。
- **RL 栏：** DRIVE 以成功轨迹的行为多样性作内在奖励，真机 OOD 成功率 64.1% 到 73.3%；EigenDEXplore 引入人类手部先验做结构化探索；PathTime-VLA 路径与时间解耦的分解式后训练。
- **生成式策略栏：** Immiscible Diffusion Policy（无标签噪声分配保留多模态）；ESP（能量评分的单步多模态生成）；RoboPace（接触感知的动作块重定时）。
- **双臂 / 灵巧 / 可变形：** ACG-Bench（8 类任务、23 组条件），AE-VLA 仿真未见组合 21.53% 对 π₀.₅ 的 2.94%，真机 39% 对最强对比 10%；CoRE 单策略多角色；OmniHOI 单目视频到灵巧手轨迹，动捕轨迹成功率 39%–89%（此前最高 31%）；OmniDex 杂乱场景抓取；绳索动态操作（Point It, Strike It）；布料 269 条真实轨迹显示结果变化比重复定位误差高 1–3 个数量级，四种仿真器均无法再现。
- **人形 / 长程 / 数据栏：** InterMimicGen、HULK、Workhorse、I-BFM 四篇人形移动操作；PACE 进度对齐上下文；Detect–Restore–Resume 在 BOSS-44 上完整任务成功率 7.6% 到 26.5%；SkillWeave 异构示范组合；BiGym 2.0、OpenViTac、TACROSS、ManiUnit 为基准与数据/硬件。
- **局限（原文自身）：** 周报只给「问题 / 创新点」摘要，多数条目（RealtimeWAM、Long-WAM、UNITAS、HULK 等）描述概括、无量化结果；机构多为「待核实」；各数字来自各论文自己的协议，不可横向比较。

## 对 wiki 的映射

- [`wiki/overview/frontier-manipulation-2026-10-05-10-09.md`](../../wiki/overview/frontier-manipulation-2026-10-05-10-09.md)
- 逐条目索引见 [frontier_manipulation_2026_10_05_10_09.md](frontier_manipulation_2026_10_05_10_09.md)

## 参考来源

- 公众号原文：<https://mp.weixin.qq.com/s/oPAUE5EvKeOfGU-Vjsynrw>
