# 3D GameDev Arena：长时程浏览器游戏开发 Agent 评测（arXiv:2611.24073）

> 一手论文来源归档（paper）

- **类型：** paper
- **英文标题：** 3D GameDev Arena: Benchmarking Coding Agents That Plan, Build, and Playtest extreme Long-Horizon Browser Games
- **中文标题：** 3D GameDev Arena：评测能够规划、构建与试玩超长时程浏览器游戏的编程智能体
- **作者：** Bizhe Bai, Jiliang Jian, Yuqing Yan, Tao Chen, Ying Wen
- **机构：** 复旦大学（Fudan University）、上海创智学院（Shanghai Innovation Institute）、西安交通大学（Xi’an Jiaotong University）、上海交通大学（Shanghai Jiao Tong University）
- **版本：** arXiv:2611.24073v1，2026-11-18，预印本（23 页）
- **论文：** https://arxiv.org/abs/2611.24073
- **全文 HTML：** https://arxiv.org/html/2611.24073v1
- **PDF：** https://arxiv.org/pdf/2611.24073
- **项目试玩：** https://baibizhe.github.io/gamearena/
- **项目页归档：** [3D GameDev Arena](../sites/3d-gamedev-arena.md)
- **核对范围：** 通读 arXiv v1 可访问 HTML 全文的正文 §1–7、参考文献与附录 A–G，逐项核对任务构造、三层评测、Table 2–4、技能化运行协议与分析限制；未验证 PDF 图像原始细节，未复现实验。
- **通读：** 已完成
- **入库日期：** 2026-11-20

## 核心摘录（论文报告，非独立复现）

1. **任务与约束（§3、Appendix B）：** 108 个从零游戏开发任务，14 类游戏类型、Medium/Hard 两档；输入文字规格与 UI/UX 参考图，从空工作区开始，要求可浏览器运行的游戏入口 `index.html`。任务不附源代码、资产或测试钩子；不能误说 108 项全为 3D，论文对任务难度和 3D 展现有分层。
2. **长时程开发流程（§3.3、Appendix F）：** 面向任务需求规划架构、收集/制作 3D 资产，按计划编排子智能体开发，经过浏览器烟测、游戏试玩、诊断和修复后产出可运行游戏。
3. **功能评测（§4.1、Appendix C）：** 编写并冻结原始验收规则，再对候选游戏适配非视觉测试探针，剔除无效项；每条有效验收标准的满足比例取等权平均，不能把它当成对全部游戏质量的评价。
4. **视觉与可玩性（§4.2、Appendix D）：** 智能体在游戏中采集关键时刻，3D presentation 以固定参考做配对评价；playability 关注操控与流程。关键时刻中论文报告 37% 自然到达、63% 辅助到达，应留意覆盖偏差。
5. **人类偏好（§4.3、Appendix E）：** 公开 Arena 进行盲测两两比较，用 Bradley–Terry 拟合相对强度；BT Score 不是任务通过率。
6. **规模与成本（摘要、Table 3）：** 各配置单任务中位工具调用 692–1067，归一化耗时约 2.65–7.11 h；单次最长试验为 14,407 次调用、约 2.13B tokens 与 608 项资产，这是极端个案，不能写成平均用量。
7. **部分结果（Table 2）：** codex-gpt-6.1-sol 功能分 64.7、BT 69.11；codex-gpt-6-astra 功能分 59.4、BT 69.24；kimi-k3 功能分 59.8、BT 25.86；zcode-glm-5.3 功能分 51.2、BT 35.78。不同模型与 harness 绑定，不能将差异归于模型本身。
8. **研究结论与边界（§6–7）：** 单看生成代码行数、上下文消耗或功能通过率，均不足以解释人类游戏偏好；浏览器调试和可玩性验证重要。尚不等于机器人/具身控制真实任务能力。

## 开源与试玩核验边界

用户提供的 GameArena URL 当前对应 GitHub Pages 仓库 `baibizhe/baibizhe.github.io` 的 `gamearena/index.html`；文件只执行外部服务器跳转（以 2026-11-20 的 master 分支为准）。无法据该跳转页验证其后台试玩、评测任务/数据、完整训练或运行源代码是否开放。未发现经核实的官方可执行仓库，**不得推断“已开源”或“确定未开源”**。

## 对 wiki 的映射

- [3D GameDev Arena：独立论文实体](../../wiki/entities/paper-3d-gamedev-arena.md)
