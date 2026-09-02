# Dual Process Motion Planning（arXiv:2609.01260）

> 来源归档（ingest）

- **标题：** Dual Process Motion Planning
- **短名：** Dual-MP
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.01260>
- **PDF：** <https://arxiv.org/pdf/2609.01260>
- **代码：** <https://github.com/verayannn/System-1-and-System-2-in-Motion-Planning>
- **机构：** 香港中文大学（深圳）；牛津大学（University of Oxford）
- **入库日期：** 2026-09-02
- **索引来源：** [具身智能小站 7 篇盘点](../blogs/wechat_embodied_station_7_papers_contact_manipulation_2026-09-02.md)
- **一句话说明：** 神经符号运动规划：经验驱动 System-1 + 符号求解器 System-2，元认知控制器动态调度。

## 开源状态（步骤 2.5，2026-09-02）

- **已开源**：`verayannn/System-1-and-System-2-in-Motion-Planning`。

## 核心摘录（面向 wiki 编译）

### 摘录 1：双系统架构

- System-1 为神经网络策略（成功轨迹训练）；System-2 含 MPC 与 CBF 在线求解器；元认知控制器选择快直觉或慢推理。
- **对 wiki 的映射：** [paper-dual-process-motion-planning](../../wiki/entities/paper-dual-process-motion-planning.md)

### 摘录 2：评测

- 多类非线性 benchmark 上规划效率、准确性与泛化性稳定增益。
- **对 wiki 的映射：** [paper-dual-process-motion-planning](../../wiki/entities/paper-dual-process-motion-planning.md)

## 当前提炼状态

- [x] 仓库已交叉核查
- [x] wiki 映射：`wiki/entities/paper-dual-process-motion-planning.md` 新建
