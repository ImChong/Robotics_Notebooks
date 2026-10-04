# Blind Grasp Reflex（官方项目页）

- **标题：** See to Reach, Feel to Grasp: Learning A Blind Grasp Reflex for Anthropomorphic Robotic Hands
- **类型：** site / project-page
- **项目页：** <https://blindgraspreflex.github.io/>
- **论文：** <https://arxiv.org/abs/2609.31323>
- **论文 HTML：** <https://arxiv.org/html/2609.31323>
- **论文 PDF：** <https://arxiv.org/pdf/2609.31323>
- **官方视频：** <https://www.youtube.com/watch?v=6EIIcab385g>
- **作者：** Alexander Alexiev、Tzu-Yuan Lin、Sang Min Kim、Ho Jae Lee、Yonghyeon Lee、Sangbae Kim
- **机构：** 麻省理工学院（MIT）、首尔大学（SNU）、延世大学（Yonsei University）
- **入库日期：** 2026-10-04
- **配套论文归档：** [See to Reach, Feel to Grasp（arXiv:2609.31323）](../papers/see_to_reach_feel_to_grasp_arxiv_2609_31323.md)
- **知识节点：** [Blind Grasp Reflex](../../wiki/entities/paper-blind-grasp-reflex.md)

## 一句话摘要

Blind Grasp Reflex 将手臂的全局接近/任务控制与灵巧手的局部接触反馈分开：手部策略只看关节位置历史及命令误差，输出手指目标和抓取就绪分数，让上层手臂在抓稳后继续搬运。

## 项目页核查（2026-10-04）

- 项目页提供系统结构图、交互式仿真抓取回放、仿真结果、真机物体照片与任务演示视频。
- 页面显示 **Code (Coming Soon)**；未列出 GitHub 仓库、可下载 checkpoint 或数据集链接。因此当前归类为 **代码待发布 / 定量结果暂不可完整复跑**，不能把演示页面当成可运行实现。
- 页面提供 12 个精选仿真抓取回放，并说明浏览器直接渲染记录，不需要在线仿真服务器；这是结果查看器，不是训练或评测代码。
- 页面强调的“blind”是部署时的**手部策略**不使用图像、物体几何、臂状态或目标末端位姿；手臂仍可由视觉、VLM 或独立规划器控制，并通过兼容接口把手送到可接触区域。

## 项目页公开结果

| 指标 | Blind Grasp Reflex | 端到端 RL 对照 |
|------|--------------------|----------------|
| YCB 静态抓取（78 个物体） | 96% | 98% |
| GraspXL 静态抓取（3,028 个物体） | 95% | 97% |
| 物体抓取过程中移动 | 92% | 8% |
| 训练区域外的物体位姿泛化 | 90% | 15% |

页面还展示同一个冻结的手部策略搭配不同手臂控制器、遮挡下抓取、VLM 条件抓放和拧灯泡等案例；这些是演示，不等价于对任意手臂控制器的普遍兼容证明。

## 关联

- 论文原始摘录：[see_to_reach_feel_to_grasp_arxiv_2609_31323.md](../papers/see_to_reach_feel_to_grasp_arxiv_2609_31323.md)
- Wiki 编译：[paper-blind-grasp-reflex](../../wiki/entities/paper-blind-grasp-reflex.md)
