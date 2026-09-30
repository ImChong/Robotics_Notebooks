# Dyna-2.1 Research Page（dyna.co/dyna-2.1）

> 来源归档（ingest）

- **标题：** Dyna-2.1: A Physical Agent for End-to-End Workflows
- **类型：** research site / company technical report page
- **官方入口：** <https://www.dyna.co/dyna-2.1>
- **公司首页：** <https://www.dyna.co/>
- **论文 / arXiv：** **无**（公司 Research 长文，2026-09-29）
- **代码：** **未开源**（页内无 GitHub / HF 链；截至 2026-09-30）
- **数据集：** **未公开**
- **机构：** Dyna Robotics
- **入库日期：** 2026-09-30
- **一句话说明：** Dyna-2.1 全栈 **Physical Agent**：半人形轮式双臂机 **Taku** + 三层模型（100 Hz 全身 RL 控制器 / 改进 **DYNA-2 WAM** / **VLM 工作流编排器**），官方称 **~1 小时无剪辑** 自主完成洗衣房全流程（洗烘、取放、折叠上架、进度记忆与部分错误恢复）。

## 开源状态（项目页核查，2026-09-30）

| 项 | 状态 |
|----|------|
| Research 正文 | 已发布（含视频与交互图） |
| Code / Weights | **确认未开源** |
| Dataset | **未公开** |
| 结论 | **闭源产业发布**；可作「长时工作流 physical agent」选型参照 |

## 页面结构（策展）

| 区块 | 内容要点 |
|------|----------|
| §1 Workflows vs tasks | 客户买的是 **整班岗位** 而非单任务；需大工作空间、可教性、推理 |
| §2 Taku + Physical Agent | 四轮转向底座 + 折叠下身 + 双 7-DoF 臂；URR 统一人/机轨迹接口 |
| §3.1 Teachability | 人数据经 URR 进策略；控制器与策略共进化；硬件改版主要重训控制器 |
| §3.2 Orchestrator | VLM 跟踪 13 决策点、文本长期记忆、低频决策 + 高频 WAM 执行 |
| §4 Forward | 客户现场部署飞轮；主张用 **长任务完成时域** 替代单 episode SR 评测 |

## 对 wiki 的映射

- 博文摘录：[`sources/blogs/dyna_2_1_physical_agent_taku.md`](../blogs/dyna_2_1_physical_agent_taku.md)
- 沉淀 **[`wiki/entities/dyna-2-1.md`](../../wiki/entities/dyna-2-1.md)**
- 前代 WAM：[`wiki/entities/dyna-2.md`](../../wiki/entities/dyna-2.md)
- 公司站：[`sources/sites/dyna-co.md`](./dyna-co.md)
