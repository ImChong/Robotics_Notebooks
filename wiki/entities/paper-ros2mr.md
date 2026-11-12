---
type: entity
project_id: ros2mr
tags:
  - paper
  - ros2
  - mixed-reality
  - motion-planning
  - visualization
  - tum
status: complete
updated: 2026-11-11
arxiv: "2611.11830"
original:
  - ../../sources/papers/ros2mr_arxiv_2611_11830.md
related:
  - ../concepts/ros2-basics.md
  - ../methods/trajectory-optimization.md
  - ./ros2-control.md
  - ./paper-notebook-armada-augmented-reality-for-robot-manipulation.md
  - ./paper-sysml-v2-ros-2-library.md
sources:
  - ../../sources/papers/ros2mr_arxiv_2611_11830.md
summary: "ROS2MR（arXiv:2611.11830）：把 YAML 序列化的 ROS 2 / MoveIt 关节轨迹转成 MR-Robotics LAB 的 JSON 回放文件，按 30 Hz 重采样（有速度用三次 Hermite，无则线性）并支持夹爪序列与 Apple Vision Pro；评测仅限数值与单元测试，无真机、头显计时或用户研究。"
---

# ROS2MR（arXiv:2611.11830）

**ROS2MR**（*ROS2MR: Connecting ROS 2 Motion Planning to Mixed-Reality Trajectory Visualization*，[arXiv:2611.11830](https://arxiv.org/abs/2611.11830)，TUM 海尔布隆校区，cs.HC）是一个工具类工作：把 ROS 2 导出的关节轨迹文件转成混合现实应用可回放的 JSON。项目页 <https://mr-lab.se.cit.tum.de/> 是需登录的网页应用，论文与首页均未给代码仓库，故本页不写 `code:`。

## 一句话定义

**作者自述：** MR-Robotics LAB 平台的一个扩展，把 YAML 序列化的 ROS 2 关节轨迹（含 MoveIt 规划）经校验、按名映射、30 Hz 重采样后写成既有 JSON 回放格式，使轨迹可在 Meta Quest 3 与 Apple Vision Pro 上脱离在线 ROS 连接查看（摘要）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MR | Mixed Reality | 混合现实，在真实环境中叠加虚拟机器人 |
| AR | Augmented Reality | 增强现实 |
| ROS 2 | Robot Operating System 2 | 机器人中间件，本文输入来源 |
| MoveIt | MoveIt motion planning framework | ROS 运动规划框架，输出关节轨迹 |
| OMPL | Open Motion Planning Library | 采样式规划算法库（相关工作提及） |
| YAML | YAML Ain't Markup Language | 导出轨迹所用的序列化格式 |
| JSON | JavaScript Object Notation | MR 应用的回放文件格式 |
| RMSE | Root Mean Square Error | 均方根误差，评测指标（式 4） |
| FD | Finite Differences | 有限差分，无速度时估计输出速度 |
| RQ | Research Question | 论文的三个研究问题 |

## 为什么重要

- 规划结果常只能在 RViz 桌面里看；该工作让学生在自己的 ROS 环境生成轨迹，再导出到头显里绕着虚拟机器人看（第 2.3 节「定位」）。
- 它把「ROS 轨迹点是带时间戳的位形，而不是播放帧」说清楚：若每个点当一帧，时长会取决于点数而非时间（第 1 章）。
- 对做规划到可视化接口的读者，这是一个小而具体的格式转换范例；它不是新规划算法，也不是新交互范式。

## 核心原理与流程总览

三阶段（3.1，图 1）：Python 导出助手生成 YAML，网页转换器校验、映射、重采样，经共享 JSON 构建器写入 Firebase，由 Unity 应用在头显回放。

```mermaid
flowchart LR
  A[ROS 2 或 MoveIt 规划] --> B[Python 导出助手]
  B --> C[轨迹 YAML]
  C --> D[网页转换器 校验与按名映射]
  D --> E[30 Hz 重采样]
  E --> F[共享 JSON 构建器]
  F --> G[Firebase 存储]
  G --> H[Unity 应用 Quest 3 或 Vision Pro]
```

关键机制（均出自论文第 3 章）：

| 环节 | 做法 | 出处 |
|------|------|------|
| 校验 | 至少 2 点；time_from_start 必须显式、严格递增；可选的速度 / 加速度 / 力矩各点覆盖不一致即拒绝 | 3.3 |
| 关节映射 | 6 个臂关节按名字重排（UR5e、KUKA joint_a1–a6、joint1–6）；夹爪通道名 gripper / finger_joint / tool / tool0 | 3.3 |
| 重采样 | 起点时间归零；N = max(2, floor(30T+0.5)+1)；前 N-1 帧在 k/30 网格，末帧精确落在 T，故末间隔可不是 1/30 s | 3.4 |
| 插值 | 各点都有速度用三次 Hermite（式 1）；否则线性（式 2）并用中心 / 前向 / 后向差分估速度；不使用加速度 | 3.4、5.5 |
| 夹爪 | 二值 0 开 1 闭，零阶保持采样，不插值 | 3.3、3.5 |
| 序列导出 | 段间位置差超 1e-4 rad 拒绝；夹爪切换处插入 1/30 s 保持点，且前段末端速度超 1e-6 rad/s 拒绝 | 3.2、3.5 |
| Vision Pro | Unity PolySpatial 加 RealityKit；Firebase 改走 REST；交互改为注视加捏合 | 3.7 |

## 评测结果（数值与单元测试）

论文只评测导出的 JSON 文件本身，不含真机、头显播放计时或用户研究（5.5）。

| 研究问题 | 设置 | 关键数字 | 出处 |
|----------|------|----------|------|
| RQ1 插值精度 | 6 关节、8 秒三次参考轨迹，稠密 / 稀疏 / 不规则三种输入，各带 / 不带速度 | 带速度时位置 RMSE 约 2.91e-5 度（三种密度相同）；不带速度时稠密 9.20e-5 度、稀疏 0.8536 度（最大 2.4243 度）、不规则 0.4293 度 | 表 3 |
| RQ2 真实样条夹具 | 两份 UR5e 样条（预设路点，非 MoveIt 生成） | 不带速度时位置 RMSE 仅 7.06e-4 度，但速度 RMSE 0.05249 度/秒、最大 1.0682 | 表 4 |
| RQ3 序列与夹爪 | 样条夹爪序列 403 点 | 转成 243 帧、约 8.067 s，开 / 闭 / 开顺序保留；0.51 s 指令推迟到 0.5333 s 出现 | 4.3 |
| 测试套件 | 31 个 JS 回归测试 + 19 个 Python 测试 | 全部通过；Python 侧用 ROS 消息桩 | 4.4 |

没有与其他转换工具或 RViz 回放的定量对比基线。

## 与其他工作对比

| 工作 | 关注点 | 与本文的区别 |
|------|--------|--------------|
| 本文 ROS2MR | ROS 2 关节轨迹离线转成 MR 回放文件 | 无在线 ROS 连接，只做格式转换与 30 Hz 重采样 |
| [ARMADA 增强现实机器人操作](./paper-notebook-armada-augmented-reality-for-robot-manipulation.md) | 用 AR 辅助机器人操作数据采集 | 面向示教与数据采集；ROS2MR 只做规划结果的可视化回放 |
| [MoveIt 2](./moveit2.md) | ROS 2 运动规划框架，自带 RViz 可视化 | 是 ROS2MR 的上游输入来源；论文并未用真实 MoveIt 规划做评测（5.2） |
| [ros2_control](./ros2-control.md) | 控制器侧的关节轨迹插值与执行 | 真实执行插值；ROS2MR 的插值仅用于显示，不等价于控制器重建（5.5） |

## 结论

**ROS2MR 是一个把 ROS 2 关节轨迹按真实时间重采样成 MR 回放文件的小工具，价值在于把「轨迹点是带时间戳的位形而非播放帧」这一点落实为可校验的转换规则。**

1. 严格校验（显式递增时间戳、速度覆盖一致）加按名映射，让 UR5e 与 KUKA 等模型共用一条导入路径（3.3）。
2. 带速度的输入用三次 Hermite 插值，位置误差在 1e-5 度量级；不带速度时稀疏输入误差可达约 0.85 度（表 3）。
3. 位置吻合不保证速度吻合：UR5e 夹具位置 RMSE 7.06e-4 度，速度最大误差却达 1.07 度/秒（表 4）。
4. 局限：评测只覆盖三次参考与预设路点样条，没有 MoveIt 规划集、真机、头显计时与用户研究，代码也未公开（5.5）。

## 工程实践

- **使用路径：** 在 ROS 脚本里用导出助手生成 YAML，上传网页应用并选虚拟机器人模型与夹爪模型（3.1）。平台有 12 款 KUKA 与 1 款 UR5e 模型（表 2）。
- **评测设计（3.6）：** RQ1 用 6 关节、8 秒三次参考轨迹，三种输入密度（401 点 50 Hz、5 点、7 个不规则点）各带 / 不带速度；RQ2 用两份 UR5e 样条夹具；RQ3 查段边界、夹爪序列与 31 个 JS 加 19 个 Python 测试。
- **复现注意：** 代码未公开（见局限），表中数字只能对照论文，本页未复现。
- **使用建议（我的判断，非论文结论）：** 若要看速度曲线，应导出带速度的轨迹；论文数据显示位置吻合不保证速度吻合（4.2）。

## 与文章说法的核对

对照公众号文章对该论文的摘要（claims 文件）：

| 文章说法 | 论文核对 |
|----------|----------|
| 将轨迹转成统一文件 | 属实：YAML 转既有 JSON 回放格式（摘要、3.1） |
| 按时间重采样供头显显示 | 属实：30 Hz，Quest 3 与 Vision Pro（3.4、3.7）；但论文自己说没测头显实际播放（5.1） |
| 保留关节与夹爪时序 | 部分：开 / 闭 / 开顺序保留；但落在网格外的指令会被推迟（0.51 s 变 0.5333 s），两采样点之间的短暂状态可能丢失（4.3、5.3）；起点延迟也被去掉（3.4） |
| 数值测试验证转换一致性 | 属实但限定：受控三次轨迹与两份 UR5e 样条，另有单元测试（3.6）；作者明说不代表任意轨迹（5.2） |
| 尚未充分测量真机同步延迟 | 措辞偏差：论文是文件回放、无在线 ROS 连接，没有「同步」可测；缺的是头显播放计时与真机执行，两者都完全没做（5.5），不只是「不充分」 |
| 操作人员使用收益未测 | 属实：无用户研究，列为未来工作（5.5、5.6） |
| 文章未提 | MoveIt 其实没有被评测：UR5e 夹具来自预设路点，不是 MoveIt 生成（3.6、5.2） |

## 局限

**性质：** 工具 / 系统类论文，核心是格式转换器，不提出新算法；评测只测导出的 JSON，不测头显画面。

**论文自己承认的（5.5）：**
- 无有代表性的 MoveIt 规划集合、无真机执行、无头显计时、无用户研究。
- 完整的导出到回放流程未对平台所有机器人模型测试；Python 测试用 ROS 消息桩，不是真实 ROS 安装。
- 小误差来自三次参考与 Hermite 同构，不能外推到任意轨迹；不用加速度，可能与控制器重建的运动不同。
- 段边界只查位置，不保证速度连续或可物理执行；力矩近似未对照真机。
- 夹爪时间受输出网格限制，可能丢短状态；Unity 播放调度未检查。

**我的补充（推测）：**
- 代码与服务入口：未见公开仓库，项目页是登录制网页应用，外部读者难以独立验证（截至 2026-11-11，项目页文档 /docs/ 未读）。
- 表 3、表 4 的误差含序列化舍入，且 RQ1 的参考函数恰为三次多项式，对论文结论的支持力度有限。

## 关联页面

- [ROS 2 基础](../concepts/ros2-basics.md)：输入轨迹所属的中间件背景。
- [轨迹优化](../methods/trajectory-optimization.md)：规划产出轨迹的上游方法。
- [ros2_control](./ros2-control.md)：关节轨迹控制器，论文 2.4 节引用其插值选项。
- [ARMADA 增强现实机器人操作](./paper-notebook-armada-augmented-reality-for-robot-manipulation.md)：同属 AR / MR 与机器人交互的相邻工作。
- [SysML v2 与 ROS 2 库](./paper-sysml-v2-ros-2-library.md)：另一篇 ROS 2 工具链类论文。

## 参考来源

- [sources/papers/ros2mr_arxiv_2611_11830.md](../../sources/papers/ros2mr_arxiv_2611_11830.md)
- 论文 PDF：<https://arxiv.org/pdf/2611.11830v1>
- 项目页：<https://mr-lab.se.cit.tum.de/>

## 继续阅读

- 论文 2.1 至 2.3 节列出的 ROS Reality、iviz、Rosen 等 MR 机器人界面工作。
- 论文 5.6 节的未来工作：非三次参考轨迹、真实 MoveIt 规划、头显计时、用户研究。
