# ViLoMan（arXiv:2609.19340）

> 来源归档（paper）

- **标题：** ViLoMan: Learning Visual-Proprioceptive Whole-Body Loco-Manipulation Skills for Humanoid Robots
- **类型：** arXiv 预印本（cs.RO）
- **作者：** Zejie Tian、Ruibing Hou、Bingpeng Ma、Börje F. Karlsson、Shiguang Shan
- **单位：** 中国科学院计算技术研究所人工智能安全重点实验室、中国科学院大学、北京智源人工智能研究院
- **arXiv：** <https://arxiv.org/abs/2609.19340>
- **PDF：** <https://arxiv.org/pdf/2609.19340>
- **项目页：** <https://viloman-anonymous.pages.dev/>
- **入库日期：** 2026-09-18；元数据与开放状态复核：2026-10-03
- **一句话说明：** 将人–物交互演示转为可执行人形轨迹，以特权教师与在线 DAgger 学习直接根据深度和本体感觉输出全身关节动作的策略。

## 开源状态

- **截至 2026-10-03：** 项目页仍为匿名补充材料页，包含方法说明、训练超参数、奖励项及演示视频；页面未列代码仓库、训练/部署代码或数据下载入口。
- arXiv 页面已公开作者和机构信息；这不代表项目代码或实验数据已公开。
- 因无官方可运行代码仓库或可下载数据，本次不新增 `sources/repos/` 条目；有可验证发布链接后再补入。

## 核心摘录

1. **问题设定：** 学习人形机器人在感知门体状态的同时协调移动、平衡与接触操作；评测覆盖不同门配置及机器人初始条件，并包含仿真和 Unitree G1 真机。
2. **可执行交互数据：** 保留演示中的人–物交互几何进行重定向，补充接近门体的运动并通过物理约束修正轨迹。
3. **特权教师与学生：** 冻结通用动作跟踪先验，以 PPO 学习特权交互残差策略；在线 DAgger 让视觉学生在自己的 rollout 中获取教师动作标签。
4. **部署策略：** 输入 4 帧 36×64 深度图和 808 维本体历史，输出 29 维关节位置动作，控制频率 50 Hz；部署时不使用参考动作、特权物体状态或中间命令。
5. **项目页复现信息：** 页面提供 teacher/student 网络与 PPO、DAgger 超参数、奖励项定义和演示视频，但截至复核日没有代码或数据下载入口。

**对 wiki 的映射**

- [paper-viloman](../../wiki/entities/paper-viloman.md)
- [Loco-manipulation 任务页](../../wiki/tasks/loco-manipulation.md)
- [10 篇技术地图](../../wiki/overview/contact-wm-10-papers-technology-map.md)
