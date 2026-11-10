# Agentic Robotics — Models, Skills, Feedback, Curriculum 对照研究（论文来源归档）

- **英文：** A Controlled Study of Models, Skills, Feedback, and Curriculum in Agentic Robotics
- **中文：** 智能体机器人中模型、技能、反馈与课程的对照研究
- **作者：** Kaiyuan Chen 等（含 Ken Goldberg）
- **论文：** https://arxiv.org/abs/2611.09030
- **全文：** https://arxiv.org/html/2611.09030v1
- **版本：** arXiv v1，2026-11-06
- **代码：** 论文 arXiv 页面未提供可确认的本研究完整实验代码入口；所用 RoboSimStudio、VABAR、MimicGen 与 open-robot-skills 属外部依赖
- **核对范围：** 阅读论文 HTML 的研究设计、策略自学习循环、四类变量、六个 Finding、实验/消融及附录关于技能格式、轨迹反馈、评测协议、成本、失败分类的内容；未重跑 631 配置的实验。
- **通读：** 已完成

## 主要核查结果

1. **研究规模：** 9 项操作任务、**631 个实验配置**、**27,620 次** coding-agent rollouts（摘要约写 28K）；不能将 rollouts 当 28K 个独立机器人。
   - 对 wiki 的映射：[对照研究实体](../../wiki/entities/paper-agentic-robotics-controlled-study.md)「实验与评测」。
2. **自学习循环：** coding agent 写初始可执行策略，仿真多实例执行，产生日志/视觉/空间/分阶段反馈，重写部分策略，并用 hold-out 任务测试及选择合适版本；不同于使用 PPO 更新单个策略网络权重。
   - 对 wiki 的映射：实体「流程总览」。
3. **六项发现：** (1) 初始能力相近的模型在自修复阶段可能分化；(2) 专家技能库对强模型是脚手架，对弱模型成边界；(3) 按语义 skill 粒度反馈比每动作步反馈更利于故障归因；(4) 对象相对/功能几何关系比原始世界系读数有用；(5) 夹持物几何、避障和接触关联规划仍是瓶颈；(6) privileged→perception 课程可能加速、也可能固化错误结构。
   - 对 wiki 的映射：实体「发现与工程结论」。
4. **反馈控制实验：** Table 2 在该组条件下，per-episode 为 25%、per-motion 36%、per-skill 52%（overall held-out 成功率），不可当成对所有任务/模型通用的固定提升。
   - 对 wiki 的映射：实体「实验与评测」。
5. **边界：** 代码型 agent 的仿真改写、几何先验和错误归因是研究对象；并不证明无需真实世界部署、安全监督或实时接触控制。
   - 对 wiki 的映射：实体「局限与风险」。
