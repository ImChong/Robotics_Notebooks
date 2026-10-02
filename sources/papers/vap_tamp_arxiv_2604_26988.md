# Robot Planning and Situation Handling with Active Perception (VAP-TAMP)

> 来源归档（ingest）

- **标题：** Robot Planning and Situation Handling with Active Perception
- **类型：** paper / tamp / active-perception / vlm
- **arXiv abs：** <https://arxiv.org/abs/2604.26988>
- **项目页：** <https://vap-tamp.github.io/vap-tamp/>
- **机构：** 宾汉姆顿分校、CMU、Ford Research、Agility Robotics 等
- **发表 / 上传：** 2026-04-28（arXiv）；**IROS 2026**
- **入库日期：** 2026-10-02
- **索引来源：** [IROS 2026 九篇获奖盘点](../blogs/wechat_iros_2026_awards_9_papers_2026-10-02.md)

## 开源状态（步骤 2.5，2026-10-02）

- **待发布：** 项目页 **Code — Coming soon**；PDF/视频可访问。

## 摘录 1：问题

- 执行期意外（门半开、物体掉落等）需检测并处理，否则长期自主性受限。

## 摘录 2：VAP-TAMP

- RGB-D + 语言目标 → 3D 地图、实例记忆、**场景图** → PDDL 规划。
- 动作前后用 VLM 验证 **前置条件/效果**；视角不足时 **主动选视点**；失败则更新状态并重规划。

## 摘录 3：评测

- 仿真服务任务 + 移动操作平台；相对基线更高成功率与可接受执行时间（项目页图表）。

**对 wiki 的映射：** [`wiki/entities/paper-vap-tamp.md`](../../wiki/entities/paper-vap-tamp.md)

## 当前提炼状态

- [x] 项目页核查
- [x] 升格 wiki
