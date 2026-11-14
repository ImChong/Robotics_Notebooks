# τ₀-VLA 项目页（tau0-vla.github.io）

> 来源归档

- **标题：** τ₀-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation
- **类型：** site
- **URL：** <https://tau0-vla.github.io/>
- **论文 PDF：** <https://tau0-vla.github.io/tau0-vla.pdf>
- **外链仓库与权重：** <https://github.com/sii-research/tau-0-vla>；<https://huggingface.co/sii-research/tau-0-vla>
- **组织：** 上海创智学院；智元机器人 Finch 团队；香港中文大学
- **入库日期：** 2026-08-19
- **核对日期：** 2026-11-14
- **通读：** 已完成
- **核对范围：** 完整通读并解析离线镜像 `sites/tau0_vla.html`，核对页面链接列表、真机长程 4 任务对比表、记忆消融 `+11.0 pp` 与 TTC 准确率增益 `+15–24 pp` 说明、以及 ARX AC One / Franka Research 3 跨本体视频展示说明。
- **一句话说明：** 官方项目页（2026-07-27 发布）：展示 13–25 步长程真机 rollout（Clean Room、Prepare Ingredients、Tomato and Egg Stir Fry、Make Milk Tea）与 ARX/Franka 跨本体视频、长程对照表以及 TTC 算力-精度曲线结论。

## 页面核对要点

1. **核心叙事：** 长程家务耗时从数秒延展到最长达 12 分钟，主要瓶颈从局部电机控制转向 **进度跟踪、后果预测与子任务规划**。多数分层 VLA 高层仍为单次前向，无法在提交动作前评估候选物理后果；τ₀-VLA 在子任务边界（稀疏且能引起可辨识视觉变化）用世界模型与价值模型执行选择性 TTC。
2. **页面披露的量化消融（项目页正文特有摘要）：**
   - 扰动演示历史构造的 **可修订执行记忆**（前进 / 回滚 / 重试）带来 **+11.0 个百分点**的 next-subtask 预测准确率提升；
   - 选择性 **TTC** 在同域及分布偏移评测上带来 **+15–24 个百分点**的 next-subtask 准确率提升；
   - 在 Make Milk Tea 上，Plan Once 与 TTC 均完成超 `91%` 任务序列，剩余失败集中于末段 **合盖与插吸管** 接触操作（TTC 将进展推至 `95.38%`）。
3. **项目页主表对照（各 10 trials/task）：** GR00T N1.7 avg `2.5%`、LingBot-VLA `0.0%`、π₀.₅ `22.5%`、τ₀-VLA 直出 `27.5%`、τ₀-VLA（Hierarchical System, Plan Once）**`45.0%`**。
4. **OOD 与 TTC 物理转译：** 未见书本初始排列 Book Organization（OOD）上，TTC next-subtask 准确率达 **74.0%**（Plan Once `50.0%`、Best-of-N `57.5%`）；闭环真机成功率：Milk Tea `5/10→7/10`、Book Organization `6/10→9/10`、Clean Room `5/10→7/10`。
5. **跨本体视频演示：** Collect Laundry（ARX AC One，5 steps）与 Tidy Makeup Table（Franka，8 steps across 3 groups）。
6. **开源状态外链：** 顶部按钮直指 `arxiv.org/abs/2608.16885`、`github.com/sii-research/tau-0-vla` 与 `huggingface.co/sii-research/tau-0-vla`。

## 对 wiki 的映射

- [τ₀-VLA 实体页](../../wiki/entities/paper-tau0-vla.md) — 综合论文与仓库工程详情
- [τ₀-VLA 论文摘录](../papers/tau0_vla_arxiv_2608_16885.md) — 论文完整结构与附录消融
- [sii-research/tau-0-vla 仓库](../repos/sii_research_tau_0_vla.md) — 仓库真实模块与已发布组件边界
