# 原始抓取：顶刊 T-RO 精选：2026上半年机器人操作学习的五项核心突破

> Agent Reach 原始正文归档（非 wiki 归纳）

- **来源 URL：** https://mp.weixin.qq.com/s/nswA-jCGC3kr9iQjhRRuXQ
- **抓取日期：** 2026-07-08
- **抓取工具：** Agent Reach v1.5.0 + `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox；`playwright==1.49.1`）
- **编译导读：** [wechat_shenlan_tro_manip_5_papers_survey.md](../blogs/wechat_shenlan_tro_manip_5_papers_survey.md)

---

（以下为抓取正文，保留 frontmatter 与结构；推广/群二维码段落已省略。）

**标题：** 顶刊 T-RO 精选：2026上半年机器人操作学习的五项核心突破  
**作者：** 深蓝具身智能  
**发表：** 2026-07-08

核心论点：2026 年上半年 IEEE T-RO 在机器人操作学习方向呈现 **数据规模化法则、三维等变表征、灵巧手物几何、无标签视频预训练、深度生成模型系统梳理** 五条主线；在规模化数据驱动下，更高级表征与生成模型正在重塑具身操作能力。

五篇论文（策展级）：

1. **Is Diversity All You Need for Scalable Robotic Manipulation?**（港大 / AgiBot / 北航等，T-RO 2026）：拆解任务/本体/演示者三维数据多样性；任务多样性 > 单任务堆量；跨本体预训练非必要；演示者多样性可能因速度多模态性干扰学习；分布去偏得 **GO-1-Pro**（+15%，等效 2.5× 预训练数据）。
2. **Canonical Policy**（浙大 / 普渡，T-RO 2026）：点云观测映射到 **规范化 3D 坐标系**，满足 **SE(3) 等变** 的策略学习；仿真 +18.0%、真机 +39.7%。
3. **DexRepNet++**（浙大 / NUS 等，T-RO 2026）：**DexRep** 手物几何与空间交互表征 + DRL；40 物体训练在 5000+ 未见物体抓取 **87.9%**；手内重定向与双手交接 +20–40%。
4. **Learning From Videos Through Graph-to-Graphs Generative Modeling**（北理工，T-RO 2026，**G3M**）：视频帧抽象为物体顶点 + 视觉动作顶点图；图到图生成预训练 → 下游策略；20% 标注达全量性能；跨本体 +33%+（CVPR 2025 姊妹版 **GraphMimic**）。
5. **A Survey on Deep Generative Models for Robot Learning From Multimodal Demonstrations**（FAIR / 英伟达等，T-RO 2026）：EBM / 扩散 / 动作值图 / GAN 与 VAE→流匹配演进；抓取/轨迹/代价学习应用与 OOD 泛化设计决策。

收束：数据正交多样性、三维等变与手物几何表征、无标签视频结构化知识、生成式决策模型四条脉络并行；场景复杂度、标注依赖与指标解读仍须审慎。
