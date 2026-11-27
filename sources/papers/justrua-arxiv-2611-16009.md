# JustRUA: A Controlled Study of GPT-6 Robot-Use Agents

- **类型：** paper / robotics
- **中文标题：** JustRUA：GPT-6 机器人操作智能体的受控拆解研究
- **论文：** [arXiv:2611.16009v1](https://arxiv.org/abs/2611.16009)（2026 年 11 月）
- **HTML 全文：** https://arxiv.org/html/2611.16009v1
- **PDF：** https://arxiv.org/pdf/2611.16009
- **作者：** Zefan Wang、Yufan Dang、Xuanwei Shi、Mingqi Gao 等
- **机构：** 清华大学、华中科技大学、北京大学、ModelBest、腾讯等
- **项目：** https://zefanw.github.io/JustRUA/
- **源码：** https://github.com/ZefanW/JustRUA
- **入库日期：** 2026-11-16
- **复核日期：** 2026-11-27
- **核对范围：** 原始论文 HTML v1 正文与附录 A–I，重点复核附录 B 控制预算、D 布局加权、E 行为审查及 H 坐标求解；参照作者项目和公开回放核对口径。
- **通读：** 已完成
- **一句话说明：** 固定 RoboDojo 测试集与基础 harness，分别消融模型调用历史、运动动作接口、经验复用与场景几何，构建更高效的 GPT-6 Astra 机械臂使用智能体。

## 原文核对摘要与 wiki 映射

### 方法与控制接口

沿用 RoboDojo 官方 L3 环境及 ARX X5 双臂；使用单次 episode 内持久的 GPT-6 Astra coding conversation 读取三路 RGB、RGB 估深、相机标定、夹指及腕力实际测量；调用绝对世界系 move_eef 目标，经 XPolicyLab Cartesian planner 转控制量，每轮最多执行前 20 个控制步，丢弃剩余部分并回传实际到达误差和夹持状态供下一次决策。文章另对示范、经验库、不同动作接口等做对照，最终配置不含自改经验库。

- **对 wiki 的映射：** [JustRUA：GPT-6 机器人操作智能体的受控拆解研究](../../wiki/entities/paper-justrua-robot-use-agents.md) — 核心原理及 Mermaid 流程。

### 实验与数值边界

在作者定义的 **111-episode** RoboDojo 对照评测子集（涵盖全部 42 任务），官方 L3 harness 29.7 → 最终 JustRUA 50.3；最终配置对应成功率 43.3%，首轮每 episode 模型调用成本下界 12.53 美元（可缓存输入全命中假设）。该分数不是未经校正的官方排行榜总分。

- **对 wiki 的映射：** 主实体 — 实验与评测、结论及误读警示。

### 开源、局限与工程取舍

- **开源状态（2026-11-16）：** [官方源码](https://github.com/ZefanW/JustRUA)；公开了完整的 XPolicyLab policy adapter；依赖 RoboDojo 仿真/资产、Depth Pro checkpoint 与 GPT-6 Astra 服务账户。
- **论文边界：** 只在仿真、单基准、单模型设置上评估，各配置对照未重复多次且不是全因子设计。完整适配器已开源，但复现还需外部 RoboDojo/XPolicyLab、Depth Pro 权重和可用模型账户。
- **对 wiki 的映射：** 主实体 — 工程实践、源码运行时序与局限。

## 原始链接

- [arXiv HTML v1](https://arxiv.org/html/2611.16009v1)
- [作者项目页](https://zefanw.github.io/JustRUA/)
- [公开代码](https://github.com/ZefanW/JustRUA)
- [官方回放与目录核验](../sites/justrua-robodojo-replay-explorer.md)
