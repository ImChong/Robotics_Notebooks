# RoboUniverse-10K: Toward the Scaling Era of Agentic Robotics via Real-to-Sim Environments

- **类型：** paper
- **英文标题：** RoboUniverse-10K: Toward the Scaling Era of Agentic Robotics via Real-to-Sim Environments
- **中文标题：** RoboUniverse-10K：通过真实到仿真环境迈向智能体机器人的规模化时代
- **作者：** Jinkun Liu、Yingze Wang、Situ Wang、Deyao Hong、Mingju Gao 等
- **机构：** 清华大学、Einsia.AI、北京大学、伊利诺伊大学厄巴纳-香槟分校、香港大学、伦敦大学学院、北卡罗来纳大学教堂山分校
- **年份：** 2026
- **arXiv：** 2611.31002v1
- **PDF：** https://arxiv.org/pdf/2611.31002
- **HTML：** https://arxiv.org/html/2611.31002v1
- **项目主页：** https://lab.einsia.ai/robo-universe/
- **源码：** https://github.com/Einsia/RoboUniverse-10K
- **数据：** https://huggingface.co/datasets/Einsia/RoboUniverse-10K
- **收录及核对日期：** 2026-11-24
- **核对范围：** 已获取 29 页 v1 PDF 与 HTML，核对正文方法、主实验、局限及附录 §7–8 的接口、数据来源和人工验收；Figure 1 已目视核对。附录 §9 的完整提示词尚未逐页核对。
- **通读：** 未完成

## 关键摘录（归纳）

1. §2 / Figures 1–4：从真实示教视频生成 10,807 个独立 Isaac Sim 环境，10,307 训练、500 留出；Franka 4,132、AgileX 6,675。构建代理用分割、深度与物理建模技能编写场景、参考控制器及验证器；独立 QA 检验保真、复位回放和判据。
2. §3 / Table 2：500 项留出任务由 10 位专家验收，单臂 200、双臂 300。作者报告最佳 85.8%，公开模型基线 2.6%–10.0%；接触后的精度、释放稳定和力控制是前沿模型的主要短板。
3. §4 / Tables 3–4：约 17K 条多轮轨迹、0.79B tokens，Qwen3.6-35B-A3B 全参数 SFT 后总体 0.8%→21.2%，单臂 2.0%→40.5%，双臂 0→8.3%。学到建模、修正、验证的闭环程序，而不是单一动作序列。
4. §6：仿真场景简化、复杂物件物性难恢复，主要覆盖短原子技能；在线 RL 是基础设施目标，不是本文已验证的 RL 训练收益。外部基准和真机绘图为迁移案例，不是大规模成功率证明。
5. 附录 §7–8：公开接口提供 RGB 和关节状态，以关节目标推进物理；参考 Franka 20 Hz / AgileX 15 Hz，均为 240 Hz 物理步频。任务预算中位数 3000 步；独立 verifier 终局二值判定。

**对 wiki 的映射：** [RoboUniverse-10K 论文实体](../../wiki/entities/paper-robouniverse-10k.md)。
