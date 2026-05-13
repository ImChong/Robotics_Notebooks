# 开源生态 / Repos

这里不是代码仓库镜像，而是开源项目与工具链的导航层。

`repos/` 的职责是：

> 按用途整理 simulation / RL frameworks / humanoid projects / retarget tools / utilities，让你知道“这个方向通常会用哪些工具和项目”。

---

## 适合谁看

适合：
- 你已经知道一个方向大概在做什么
- 现在想找对应的开源项目、工具链、训练框架或底层库

不适合：
- 你还没搞懂概念本身是什么
- 你想看完整安装教程（那应回到官方 docs / repo）

---

## 快速入口

| 你的目标 | 从这里进入 |
|---------|-----------|
| 想找仿真平台 | [Simulation](simulation.md) |
| 想找 RL 训练框架 | [RL Frameworks](rl-frameworks.md) |
| 想找 humanoid / legged 项目 | [Humanoid Projects](humanoid-projects.md) |
| 想找动作重定向工具 | [Retarget Tools](retarget-tools.md) |
| 想找底层工具与 utilities | [Utilities](utilities.md) |
| 想找操作 / 抓取感知 SDK 与数据接口 | [Manipulation Perception](manipulation-perception.md) |

---

## 当前主线怎么对应到 repos/

### 仿真 / 训练平台主线
如果你在看：
- Isaac Gym / Isaac Lab
- MuJoCo
- legged_gym
- RL locomotion

建议从这里进入：
- [Simulation](simulation.md)
- [RL Frameworks](rl-frameworks.md)

### 人形 / 足式项目主线
如果你在看：
- Unitree
- humanoid locomotion
- legged robot projects

建议从这里进入：
- [Humanoid Projects](humanoid-projects.md)

### 控制 / 优化工具链主线
如果你在看：
- Pinocchio
- Crocoddyl
- TSID / WBC
- trajectory optimization

建议从这里进入：
- [Utilities](utilities.md)

### 动作迁移 / 模仿学习主线
如果你在看：
- Motion Retargeting
- Imitation Learning
- skill transfer

建议从这里进入：
- [Retarget Tools](retarget-tools.md)

### 操作 / 抓取感知主线
如果你在看：
- bin picking、平行夹爪抓取位姿估计
- 深度点云上的稠密抓取与跨帧跟踪

建议从这里进入：
- [Manipulation Perception](manipulation-perception.md)

---

## 当前判断

`repos/` 后续的重点不是简单加更多 GitHub 链接，而是：

1. 让每个分类更明确它解决什么问题
2. 让 wiki 主线能自然对应到 repo 入口
3. 区分“训练框架”“底层库”“完整项目”“工具类 repo”
