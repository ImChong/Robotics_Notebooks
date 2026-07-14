# humanoid_motion_control_know_how

> 来源归档（ingest）

- **标题：** 人形机器人运动控制 Know-How（飞书公开文档）
- **类型：** paper / course-map
- **来源：** 飞书公开文档 <https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87>
- **入库日期：** 2026-04-18
- **最后更新：** 2026-07-14
- **一句话说明：** RoboParty 系统化人形运动控制课程地图：趋势、双学习路线、问题框架、Model-based 七段主链、Learning-based 方法族与 BFM 三线；每法「原理 / 代码 / 局限性」。

## 来源上下文

- **文档风格：** 课程地图 / 方法总纲，非单篇论文
- **抓取：** 2026-07-14 经 Agent Reach v1.5.0 + Jina Reader 部分正文 → [raw 摘录](../raw/feishu_humanoid_motion_control_know_how_2026-07-14.md)
- **完整目录：** [know-how.md](../notes/know-how.md)、[resources/knowhow](../../resources/knowhow/人形机器人运动控制Know-How.md)

## 文档结构（与 wiki 一一对应）

详见 **[humanoid-motion-control-know-how-technology-map](../../wiki/overview/humanoid-motion-control-know-how-technology-map.md)** 全主题索引表。

### 宏观

- 发展趋势 → `wiki/overview/humanoid-motion-control-trends.md`
- 学习路线（传统 / RL）→ `roadmap/depth-classical-control.md`、`roadmap/depth-rl-locomotion.md`
- 问题解决思路 → `wiki/concepts/modeling-and-solving-for-control.md` 等

### Model-based 主链

OCP → LIP/ZMP → SLIP/VMC → WBD+WBC/TSID → SRBD+Convex MPC+WBC → CD+NMPC+WBC → 状态估计

### Learning-based

RL、Teacher-Student+DAgger、DreamWaQ、PIE、Attention 落足、Retarget、DeepMimic、AMP、BFM（Zero / SONIC / 多技能 TS）

## 对 wiki 的映射

- **图谱父节点：** [humanoid-motion-control-know-how-technology-map.md](../../wiki/overview/humanoid-motion-control-know-how-technology-map.md)
- **Query 摘要：** [humanoid-motion-control-know-how.md](../../wiki/queries/humanoid-motion-control-know-how.md)
- **工程 Know-How（部署）：** [overview/humanoid-motion-control-know-how.md](../../wiki/overview/humanoid-motion-control-know-how.md)（IMU/热管理等，与飞书课程图不同页）

## 当前提炼状态

- [x] 全主题树 → 独立 wiki 节点（2026-07-14）
- [x] 飞书部分正文抓取归档
- [x] PIE、DreamWaQ 一手论文 source 补链
- [~] 飞书逐节伪代码与完整正文待作者公开 API 或手动补充
