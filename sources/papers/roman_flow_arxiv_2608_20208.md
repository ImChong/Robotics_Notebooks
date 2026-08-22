# RoMAN-Flow: Taming Autoregressive Normalizing Flows for Offline Reinforcement Learning in Robotic Manipulation

> 来源归档（ingest）

- **标题：** RoMAN-Flow: Taming Autoregressive Normalizing Flows for Offline Reinforcement Learning in Robotic Manipulation
- **短名：** RoMAN-Flow
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.20208>
- **PDF：** <https://arxiv.org/pdf/2608.20208>
- **项目页：** <https://github.com/konnyaku28/RoMAN-Flow>
- **代码：** <https://github.com/konnyaku28/RoMAN-Flow>
- **入库日期：** 2026-08-22
- **索引来源：** [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md)（<https://mp.weixin.qq.com/s/EmC4gNgcQdPX34vxy-qSVQ>）
- **一句话说明：** 见下方摘录与 wiki 映射。

## 开源状态（步骤 2.5）

- **已开源**：[`konnyaku28/RoMAN-Flow`](https://github.com/konnyaku28/RoMAN-Flow) 含 LIBERO/RoboMimic 训练评测脚本；HF 权重 [`wangshaoxuan/RoMAN-Flow`](https://huggingface.co/wangshaoxuan/RoMAN-Flow)。

## 核心摘录（面向 wiki 编译）

### 摘录 1：离线 RL + AR-NF

- sampling-free advantage-weighted likelihood 提高高优势离线动作似然；部署蒸馏为一步动作生成器。

**对 wiki 的映射：** offline-rl、normalizing-flow

### 摘录 2：结果

- 多仿真操作基准与真机 competitive 策略表现，显著降低推理延迟。

**对 wiki 的映射：** 操作策略部署

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-roman-flow.md`](../../wiki/entities/paper-roman-flow.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链
