# Genie Studio Agent（智元官方平台与发布归档）

> 来源归档（官网产品页 + 官方新闻稿）

- **标题：** AGIBOT Launches Genie Studio Agent to Enable Scalable Robot Deployment / Genie Studio 一站式开发平台
- **类型：** site / official-article
- **机构：** 智元机器人（AgiBot）
- **官方入口：**
  - Genie Studio Agent 发布稿（2026-04-13）：<https://agibot.com/news/newsroom/agibot-launches-genie-studio-agent-to-en-59>
  - Genie Studio 产品主页：<https://genie.agibot.com/geniestudio>
  - Genie Studio 首发稿（2025-04-21）：<https://www.agibot.com.cn/news/company/genie-studio-61>
- **入库日期：** 2026-11-21
- **核对范围：** 通读智元英文新闻稿 <https://agibot.com/news/newsroom/agibot-launches-genie-studio-agent-to-en-59>、Genie Studio 产品主页 <https://genie.agibot.com/geniestudio> 及中文首发稿 <https://www.agibot.com.cn/news/company/genie-studio-61> 全文（含四大模块、GDK 开发组件、零代码编排、仿真预演、真机 RL 自优化、华天科技半导体封测晶圆搬运案例）；经官网与 GitHub 检索确认 Genie Studio Agent 本身为商业闭源平台，无公开源码仓。
- **通读：** 已完成
- **开源状态：** **商业闭源平台**（Genie Studio Agent 编排器与 GDK 平台自身未开源，其底层生态对接开源的 AgiBot World 数据集、Genie Sim 评测工具及 GO-1/RDT/Pi0 等模型）
- **一句话说明：** 智元面向产业现场部署的零/低代码机器人应用编排与全生命周期交付平台：构建在 Genie Studio（采-训-测-推）与 GDK 之上，通过拖拽式工作流、3D 仿真预演、真机强化学习自优化和全链路监控，推动具身智能从定制项目向可复制产线交付演进。

## 官方原文要点

1. **从 Genie Studio 到 Genie Studio Agent：** 2025-04 发布的 Genie Studio 解决「数据采集、模型训练（集成 RDT/Pi0/OpenVLA/GR00T/GO-1）、仿真评测（6,000–8,000+ 物体资产）、一键部署（GDK + 模型编译器）」；2026-04 发布的 **Genie Studio Agent** 进一步针对工厂与车间落地瓶颈，把部署态抽象为面向非编程人员的零代码编排与全生命周期基础设施。
2. **四大核心能力（2026-04-13 发布稿）：**
   - **No-code workflow orchestration：** 将视觉感知、运动控制、导航、VLA 模型与 RL 工具链封装为可拖拽节点和行业模板。
   - **Simulation-first deployment：** 集成 3D 重建与仿真，上线前先虚拟验证任务执行、路径规划与交互异常。
   - **Real-world RL for continuous optimization：** 在产线运行期结合力控与视觉反馈持续用 RL 优化抓取/放置精度。
   - **End-to-end monitoring & proactive management：** 统一可视化监控状态、数据与异常预警。
3. **产业验证案例：** 与 **华天科技（Huatian Technology）** 合作部署于半导体封测晶圆搬运流程，串联高精度位姿调整、复杂环境导航、力控抓取与 RL 驱动放置。

## 对 wiki 的映射

- [Genie Studio Agent](../../wiki/entities/genie-studio-agent.md)
