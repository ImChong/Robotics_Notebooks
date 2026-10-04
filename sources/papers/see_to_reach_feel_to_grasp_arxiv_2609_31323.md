# See to Reach, Feel to Grasp: Learning A Blind Grasp Reflex for Anthropomorphic Robotic Hands（arXiv:2609.31323）

> 来源归档（paper）

- **标题：** See to Reach, Feel to Grasp: Learning A Blind Grasp Reflex for Anthropomorphic Robotic Hands
- **类型：** paper / dexterous grasping / proprioception / manipulation
- **arXiv：** <https://arxiv.org/abs/2609.31323>
- **HTML：** <https://arxiv.org/html/2609.31323>
- **PDF：** <https://arxiv.org/pdf/2609.31323>
- **项目页：** <https://blindgraspreflex.github.io/>
- **官方视频：** <https://www.youtube.com/watch?v=6EIIcab385g>
- **作者：** Alexander Alexiev、Tzu-Yuan Lin、Sang Min Kim、Ho Jae Lee、Yonghyeon Lee、Sangbae Kim
- **机构：** 麻省理工学院（MIT）、首尔大学（SNU）、延世大学（Yonsei University）
- **版本：** arXiv v1，2026-09-25 提交
- **一句话说明：** 将臂部全局到达与手部本体感觉抓取分离；20-DoF 手部策略用关节传感历史适应接触，并以学习到的抓取分数协调上层手臂何时继续操作。

## 项目与开放状态核查（2026-10-04）

- 官方项目页已打开核对，页面标注 **Code (Coming Soon)**，暂未链接 GitHub 仓库、模型权重或可下载数据集。
- 项目页提供交互式仿真记录和演示视频，但没有运行或重训策略的入口；完整复现实验当前受代码未发布限制。
- 论文的硬件量化证据仍是定性测试：29 个不同真实物体均有成功抓取展示，但每个物体没有重复试验与统计成功率。

## 核心摘录

1. **模块化接口：** 手臂控制器输出掌心目标位姿与抓取使能命令；手部策略输出 20 个手指关节目标及 0–1 的抓取分数。手臂读取抓取分数决定何时进入抬升或后续操作。
   **对 wiki 的映射：** [Blind Grasp Reflex](../../wiki/entities/paper-blind-grasp-reflex.md)「方法栈与闭环」。

2. **盲手策略输入：** 学生策略接收五帧关节位置与关节测量值相对上一条命令的误差历史，共 200 维；不输入图像、物体几何、手臂状态、力矩、力或触觉阵列。接触使手指运动受阻后，命令与编码器反馈之间的残差可为策略提供隐式接触线索。
   **对 wiki 的映射：** [Blind Grasp Reflex](../../wiki/entities/paper-blind-grasp-reflex.md)「核心机制」「常见误区」。

3. **训练与控制：** 仿真中先用 PPO 训练特权教师，再用 DAgger 将其蒸馏为前馈学生；奖励由手内接触覆盖、抬升高度和稳定持握组成。手策略 11.9 Hz，底层关节控制 1 kHz。
   **对 wiki 的映射：** [Blind Grasp Reflex](../../wiki/entities/paper-blind-grasp-reflex.md)「训练和部署接口」。

4. **结果及边界：** 20-DoF Robotis HX5-D20-MRT 手安装在 7-DoF Flexiv Rizon 4 臂上。仿真报告 YCB 96%、GraspXL 95%、动态物体测试 92%、训练区域外物体位姿泛化 90%；29 个真机物体是无重复次数的定性测试。
   **对 wiki 的映射：** [Blind Grasp Reflex](../../wiki/entities/paper-blind-grasp-reflex.md)「实验与评测」「局限与风险」。

## 相关一手资料

- [官方项目页](https://blindgraspreflex.github.io/) — 交互式仿真回放、系统介绍和视频
- [arXiv HTML 正文](https://arxiv.org/html/2609.31323)
- [arXiv PDF](https://arxiv.org/pdf/2609.31323)
- [项目演示视频](https://www.youtube.com/watch?v=6EIIcab385g)
