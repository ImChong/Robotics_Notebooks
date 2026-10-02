# Neuro-Symbolic Learning for Long-Horizon Task Planning Under Complex Logical Constraints (iFlax)

> 来源归档（ingest）

- **标题：** Neuro-Symbolic Learning for Long-Horizon Task Planning Under Complex Logical Constraints
- **类型：** paper / task-planning / neuro-symbolic / mobile-manipulation
- **arXiv abs：** <https://arxiv.org/abs/2606.06877>
- **项目页：** <https://sairlab.org/iflax/>
- **机构：** 纽约州立大学布法罗分校、卡内基梅隆大学等
- **发表 / 上传：** 2026-06-05（arXiv）
- **入库日期：** 2026-10-02
- **索引来源：** [IROS 2026 九篇获奖盘点](../blogs/wechat_iros_2026_awards_9_papers_2026-10-02.md)

## 开源状态（步骤 2.5，2026-10-02）

- **待发布：** 项目页有论文/视频；**截至入库日 `sair-lab/iflax` 等公开代码仓不可访问**。

## 摘录 1：问题

- 长时程规划在大量物体与逻辑约束下搜索空间爆炸；神经符号剪枝存在 train–test 搜索空间不一致（exposure bias）。

## 摘录 2：iFlax

- 物体重要性学习建模为 **双层优化**：上层神经网络评分，下层在剪枝空间 **符号规划**。
- **Repair / Restart / Rollback（3R）** 为上层提供可靠反馈。

## 摘录 3：评测

- MazeNamo 等基准：相对 Flax **失败率 −80.04%**、规划时间 **−57.14%**（论文报告）。
- Spot 四足移动操作仿真与真机验证。

**对 wiki 的映射：** [`wiki/entities/paper-iflax.md`](../../wiki/entities/paper-iflax.md)

## 当前提炼状态

- [x] 项目页核查
- [x] 升格 wiki
