# RAPID（arXiv:2609.21767）

> 来源归档（paper）

- **标题：** Scaling Vision-Language Reward Learning for Robot Manipulation in Parallel Simulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.21767>
- **PDF：** <https://arxiv.org/pdf/2609.21767>
- **代码：** https://github.com/rapid-vlm/rapid-vlm-rl
- **入库日期：** 2026-09-21
- **一句话说明：** GPU 并行 rollout + 单请求偏好标注 + 代表性图像采样 + 自动奖励稳定化；5 个 IsaacLab Franka 任务平均训练 9.18h→3.13h，全组件 1.15h、896 次 API 调用。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-21）
- 项目页/arXiv 截至入库日的可运行代码链接核查结论见上。

## 核心摘录

1. **策展档位：** 深读（[具身小站 10 篇盘点](../blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md)）
2. **机构：** （待论文正式披露）
3. **导读要点：** VLM 奖励 RL 的瓶颈常在标注吞吐而非策略网络；RAPID 把 rollout、标注、采样与更新统一调度。

## 对 wiki 的映射

- [paper-rapid-vlm-rl](../../wiki/entities/paper-rapid-vlm-rl.md)
- [10 篇技术地图](../../wiki/overview/contact-rich-sim-10-papers-technology-map.md)
