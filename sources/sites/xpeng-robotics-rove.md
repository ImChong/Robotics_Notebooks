# ROVE 项目页（xpeng-robotics.github.io/rove）

> 来源归档

- **标题：** ROVE: Unlocking Human Interventions for Humanoid Manipulation via Reinforcement Learning
- **类型：** site（项目页 + 实验视频）
- **URL：** <https://xpeng-robotics.github.io/rove/>
- **论文：** <https://arxiv.org/abs/2606.17011>
- **机构：** XPENG Robotics；复旦大学；香港中文大学；上海交通大学
- **入库日期：** 2026-06-18
- **一句话说明：** 小鹏机器人官方页：人机闭环采集 → 阶段感知标注 → OVE critic → advantage-conditioned VLA 提取；擦白板/面包入吐司机/螺丝安装真机视频；与 SFT、HG-DAgger、Filtered BC、RECAP 对比及三轮迭代曲线。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Abstract | 混合质量 rollout/干预/恢复轨迹；OVE 优先高价值行为；跨 embodiment 人类视频 |
| Method | 人机闭环采集、三阶段分解、OVE、advantage-conditioned policy extraction |
| Results | 两主任务成功率；基线对比；三轮迭代提升 |
| Value estimation | 人类经验与 OVE 对 value 曲线的影响 |
| Recovery | 部署时重试插入、补擦等恢复行为视频 |
| One More Thing | 螺丝安装高精度任务 |
| Citation | BibTeX |

## 对 wiki 的映射

- 主实体：[ROVE（论文实体）](../../wiki/entities/paper-rove-humanoid-vla-intervention.md)
- 论文摘录：[rove_arxiv_2606_17011.md](../papers/rove_arxiv_2606_17011.md)
