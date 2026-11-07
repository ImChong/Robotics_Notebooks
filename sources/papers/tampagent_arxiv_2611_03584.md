# TAMPAgent: Agentic Belief-Space Planning for Mobile Manipulation

- **中文标题：** TAMPAgent：面向移动操作的信念空间智能体规划
- **类型：** paper
- **作者：** Utkarsh A. Mishra、Alperen Degirmenci、Yongxin Chen、Danfei Xu、Tucker Hermans、Caelan Garrett
- **机构：** 英伟达研究院、佐治亚理工学院、犹他大学
- **论文：** <https://arxiv.org/abs/2611.03584>（v1，2026-11-02）
- **全文：** <https://arxiv.org/html/2611.03584v1>
- **项目页：** <https://umishra.me/TAMPAgent/>
- **项目页归档：** [TAMPAgent](../sites/tampagent.md)
- **核对日期：** 2026-11-07
- **核对范围：** v1 正文第 1–7 节及附录 A–H：形式化、逐步案例、任务与子任务、G1 结果、硬件、系统服务、沙箱与基线适配；项目页资源开放状态。
- **通读：** 已完成
- **代码状态：** 项目页 Code soon，尚未提供可运行官方仓库或评测数据下载入口。

## 核心摘录与 wiki 映射

以下为基于原文的归纳，不转存全文。

1. **混合信念（§2–3）**：未知变量由均匀先验、语言假设转为工具认证的掩码、RGB-D、抓取与轨迹。语言信念的风险、成本由智能体隐式权衡，并非完整概率信念树求解。
   - 映射：[核心原理](../../wiki/entities/paper-tampagent.md#核心原理)。
2. **首动作认证（§3.2、附录 A/F）**：整段计划编译约束，跨步骤共享抓取等变量；每轮绑定下一动作的必要参数，后续变量可保留语言假设。观察失败、IK 不可达分别触发假设或几何计划修订。detect 的目标可见性在到达视点后确认。
   - 映射：[流程总览](../../wiki/entities/paper-tampagent.md#流程总览)。
3. **比较协议（§4、附录 B–D/H）**：RB-Y1 九项厨房任务每方法每任务 20 回合，完整成功率均值 77.2%，CaP 57.2%、GaP 61.1%；T9 单独测试跨尝试记忆学习。T11–T13 每方法每任务 5 回合，三方法均为 5/5、5/5、4/5。
   - 映射：[实验与评测](../../wiki/entities/paper-tampagent.md#实验与评测)。
4. **真实边界（§4/7、附录 E–H）**：RB-Y1 展示三类任务；真机 G1 移开玩具狗后，放置未获认证即结束，未完成抓罐。真机持物由智能体观察后声明，门/抽屉和底座规划配置与仿真不同；保证依赖感知与工具，动作域固定。
   - 映射：[工程实践](../../wiki/entities/paper-tampagent.md#工程实践)、[局限与风险](../../wiki/entities/paper-tampagent.md#局限与风险)。
