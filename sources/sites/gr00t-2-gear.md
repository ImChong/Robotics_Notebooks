# NVIDIA GEAR — GR00T 2 官方项目页（来源归档）

- **标题：** GR00T 2: Scaling Robot Foundation Models for Generalization, Dexterity, and Adaptation
- **项目页：** <https://research.nvidia.com/labs/gear/gr00t-2/>
- **论文：** <https://research.nvidia.com/labs/gear/gr00t-2/assets/gr00t2_whitepaper.pdf>（独立论文归档见 [gr00t2_whitepaper_2026.md](../papers/gr00t2_whitepaper_2026.md)）
- **机构：** 英伟达（NVIDIA）
- **发布日期：** 2026-11（项目页）
- **入库日期：** 2026-11-10
- **代码：** **待发布**；项目页顶部明确写 `Code Available in December`，截至入库日无 GR00T 2 对应公开代码入口
- **数据：** [Human-1M-100K（Hugging Face）](https://huggingface.co/datasets/nvidia/PhysicalAI-Human-1M-100K)（项目页声明 10 万小时人类视频公开；具体卡片权限/许可未单独核验）
- **核对范围：** 2026-11-10 阅读 NVIDIA GEAR 项目页全部可抓取的文字：介绍、架构、三类预训练数据、72 任务人类视频规模消融、X-Adapt 三条迁移路径、双灵巧手、50 任务零/少样本适配、真机 RL、Agentic WAM API、visual steering、Turbo/SONIC/G1、Sim2Real、real-to-sim 评测与参考引文。网页中的嵌入视频与动态图表未逐帧观看；白皮书全文及附录不属于本条核对范围，另见论文归档。
- **通读：** 已完成

## 经项目页核对的机制

- **输入 → 输出：** 最近的视频、语言指令与机器人状态 → 联合去噪得到未来视频与动作 chunk；动作执行后重新观测和预测。用本体专属的小头把共享的表示映射到不同执行器。
- **训练混合：** 人类第一视角视频 402,730 小时；手持 UMI 10,184 小时、2,897 项任务；机器人遥操作 23,500 小时，含平行夹爪、灵巧手与 Unitree G1。
- **跨本体：** 人类示范 → Dexter；手持 UMI → YAM；可穿戴 DexUMI → 双手灵巧操作；目标机器人仍可提供与任务无关的本体数据，不能理解成完全不需要目标机器人数据。
- **少样本：** Dexter-50 零样本 40%、1-shot 71%、5-shot 95%；这些任务在 A.3 中明确按 5-shot 至少达 80% 成功的条件筛选，不能外推到随机新任务；单 RTX 5090 约 30 分钟 LoRA 适配。40/50 项至少成功一次是另一口径。
- **Agentic：** 推理代理拆解与核验任务，经 WAM API 调用 GR00T 2，必要时使用 CaP 处理故障恢复；三种玩具装配分别为 98/100、97/100、97/100。
- **视觉指引：** 规划器标注夹取点与放置点，YAM 短步「what-and-where」成功率从仅语言 40% 到 92%；不提供最后姿态或具体运动路径信息。
- **Turbo 全身控制：** 2B Turbo 在 G1/Jetson Thor 上约 124 ms 预测一次、生成 1.6 秒潜在全身动作段，SONIC 以 50 Hz 解码，机器人关节 PD 以 1000 Hz 执行；124 ms 是单次预测延迟，不是 50 Hz 控制周期。
- **真实/仿真：** 25 个 YAM 任务上的 14 个策略表现对比，仿真与真机成功率 Pearson 0.92、Spearman 0.89；相关不等于对每项任务的准确保证。

## 关键边界

1. 项目页为作者自报的实验结果；未在本库复现。
2. **模型代码计划 2026 年 12 月提供**，不能把既有 [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) 的 N1.7 实现当成 GR00T 2 源码。
3. **2026-11-10 更新：** 已直接阅读官方白皮书全部可提取正文（§1–§5）及附录 A–K；见独立 [白皮书归档](../papers/gr00t2_whitepaper_2026.md)。本条作为项目网站独立来源归档，保留其网页本身的通读范围；白皮书表格与局限以独立归档优先。

## 对 wiki 的映射

- [GR00T 2 项目实体](../../wiki/entities/paper-gr00t-2.md)
