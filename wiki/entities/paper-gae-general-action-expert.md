---
type: entity
tags:
  - paper
  - humanoid
  - teleoperation
  - motion-tracking
  - reinforcement-learning
  - unitree-g1
  - westlake
  - westlake-robotics
status: complete
updated: 2026-10-10
arxiv: "2609.34233"
institutions: [westlake-robotics, westlake]
related:
  - ../tasks/teleoperation.md
  - ../tasks/humanoid-locomotion.md
  - ../concepts/motion-retargeting.md
  - ./paper-teleopit.md
  - ./paper-twist2.md
  - ./westlake-robotics.md
  - ./westlake-titan-o1.md
sources:
  - ../../sources/papers/gae_general_action_expert_arxiv_2609_34233.md
  - ../../sources/sites/gae-general-action-expert.md
  - ../../sources/sites/wlrobo-gae-download.md
summary: "GAE（General Action Expert）：万小时人类动作、特权生成器与可部署执行器两阶段训练，以延迟条件预判实现 G1/O1 实时全身遥操作；西湖机器人官网以「通用小脑 / 身外化身系统」提供 V1.2.0 闭源商用安装包，算法代码与权重未开源。"
---

# GAE：General Action Expert for Real-Time Humanoid Teleoperation

## 一句话定义

**GAE** 是让人形机器人实时模仿操作员全身动作的运动控制框架：离线用生成器清洗并适配人类动作，执行器直接接收人体动作，在部署时按实际延迟预判。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GAE | General Action Expert | 本文全身遥操作框架；不同于 Geometric Autoencoder |
| SMPL | Skinned Multi-Person Linear Model | 统一人体动作表示及机器人比例代理骨架 |
| PPO | Proximal Policy Optimization | 生成器与执行器的策略训练算法 |
| RoPE | Rotary Position Embedding | 通过人体动作 token 的时间索引偏移指定预判时域 |
| PD | Proportional-Derivative | 低层执行策略给出的目标关节位置 |
| SR | Success Rate | 整段动作跟踪成功率，非单帧误差 |
| TRT | TensorRT | 官方机器人端安装包按 CUDA / TensorRT 版本分发，面向 Jetson 机载推理 |
| API | Application Programming Interface | 商用客户端的 WebSocket / HTTP 控制接口，用代码推送骨骼或关节帧 |

## 为什么重要

对全身遥操作，**动作覆盖广**与**人机同步快**是两个不同难题。直接拿嘈杂、形态不匹配的人类动作，在强域随机化下训练部署策略，会把参考质量和动力学扰动混在同一个优化问题里。GAE 把它们拆开，再把通信与执行延迟显式作为控制输入。

## 核心信息

| 项 | 内容 |
|----|------|
| 论文 | [arXiv:2609.34233](https://arxiv.org/abs/2609.34233)，2026-09-28 |
| 作者机构 | 西湖机器人（Westlake Robotics）、西湖大学（Westlake University） |
| 动作数据 | 视频 / 动画 / 动捕，统一 SMPL 后镜像、拼接、上下身重组；超过 **1 万小时** |
| 模型与仿真 | 生成器、执行器均约 **852M 参数**；Isaac Lab |
| 输入 / 输出 | 本体状态 + 人体动作目标（及延迟）→ 各关节目标位置 → 低层 PD |
| 控制频率 | 论文所述策略 **50 Hz**，一帧预判为 **20 ms** |
| 真机 | Unitree G1；Westlake O1 经形态专用微调 |
| 时间线 | 2025-09-29 「GAE身外化身系统」首次公开发布（官网 / 项目页自报）→ 2026-09-28 arXiv v1（截至 2026-10-10 仅 v1） |
| 官方定位 | 西湖机器人官网「具身大模型」中的**通用小脑**，与「大脑 LM-VLM」组成大小脑架构；宣传语「身外化身 如影随形 / 全球首个实现一脑多形的准AGI控制系统」（自报） |
| 开放程度 | [官网下载页核查](../../sources/sites/wlrobo-gae-download.md)（2026-10-10）：可下载 **V1.2.0 闭源商用安装包**（Windows 客户端 + 机器人端 `.run`），需购买并配 USB 加密狗与密钥；**算法代码、训练代码、权重、数据均未开源** |

## 方法：先生成可行轨迹，再训可部署控制器

```mermaid
flowchart TB
  data["视频 / 动画 / 动捕"] --> smpl["统一 SMPL 与增强"]
  smpl --> fit["机器人比例形态拟合"]
  fit --> gen["特权生成器：干净仿真跟踪"]
  gen --> target["物理可行机器人轨迹：奖励目标"]
  smpl --> exe["执行器：原始人体动作输入"]
  target --> exe
  exe --> deploy["域随机化 + 延迟条件预判 → 真机"]
```

- **生成器**使用特权状态及未来目标帧，在干净仿真中跟踪形态拟合参考，rollout 得到机器人可行轨迹。
- **执行器**使用机器人可部署本体观测与人体动作目标；生成轨迹只用于计算跟踪奖励。课程式域随机化逐步加入动力学差异、观测噪声和外力。部署时无需在线运行生成器或机器人轨迹重定向，但人体动作仍须被观测并整理为策略所需格式。
- **同步**：真实人体目标到机器人有端到端延迟 `τ`。执行器学习 `π(a | 本体状态, 延迟人体目标, τ)`，训练时对齐当前人体动作的奖励；RoPE 时间索引偏移表示预判时域，部署可依据测得延迟调节。预判是**动作趋势估计**，不能消除网络抖动或凭空获知不可预测的人体动作。

## 实验与评测

| 指标（论文 Table I） | SONIC | GAE | 解读 |
|------|------:|------:|------|
| Easy 序列 SR | 95.4% | 97.3% | 简单动作差距较小 |
| Medium 序列 SR | 85.2% | 96.9% | 中等动态动作优势扩大 |
| Hard 序列 SR | 77.7% | 93.9% | 跳跃、爬行、起身等优势明显 |

每组约 50 段训练集外序列；**SR 要求整段运动所有选定关键点始终在参考轨迹 50 cm 内**。Hard 集把预判从 0 增到 5 帧（0–100 ms），SR 从 **93.9%** 到 **92.1%**，关键点位置误差从 **0.045 m** 增到 **0.056 m**：同步改善存在精度代价。真机两台 G1 的 0 与 4 帧（80 ms）对照显示可见延迟降低；并展示行走、单脚、跪蹲、爬行和灵巧手交互。O1 的迁移是**微调后**，不是零样本。

## 与其他工作对比

- [SONIC](../methods/sonic-motion-tracking.md)：论文选择的全身跟踪基线；GAE 在高动态序列上的 SR 更高，但各分项误差并非全优。
- [Teleopit](./paper-teleopit.md)：同属西湖相关的遥操作研究；Teleopit 强调 PICO VR + 灵巧手 / 视点 / 录制的系统栈，GAE 聚焦大规模全身运动先验和时延条件控制。
- [TWIST2](./paper-twist2.md)：同为全身遥操作参考，可比较输入设备、策略目标、时延处理和代码开放程度。

## 结论

**GAE 的主要可复用点，是把嘈杂人体参考的“物理可行化”、真机鲁棒性训练和遥操作延迟补偿分成可分别验证的环节。**

1. 先用生成器生成可行轨迹，再用课程域随机化训执行器；不要把原始噪声参考与强扰动一次性叠加。
2. 执行器部署输入仍是人体动作，不依赖在线运行特权生成器；其“无需在线重定向”不等于无需人体感知与格式转换。
3. 用序列级 SR 和关键点误差一起验收：高动态动作成功率提升，不代表每一个姿态 / 角速度指标都更优。
4. 预判时域按实测端到端延迟设置；100 ms 预判下 Hard SR 仍为 92.1%，但位置误差有增幅。
5. 跨机器人适配需微调；没有 RGB / LiDAR 外感知，复杂地形与障碍交互仍受限。

## 官方产品与开放状态

论文之前，GAE 已作为商用「GAE身外化身系统 / GAE Emanation」发布（2025-09-29，自报）。2026-10-10 下载页与产品手册显示的部署链路：

```mermaid
flowchart LR
  op["操作者输入：动捕 / PICO VR / 键鼠 / API"] --> client["Windows 客户端 V1.2.0（加密狗鉴权）"]
  client -->|"单机 / 远程 / 群控 ≤10 台"| robot["机器人端安装包：Jetson + CUDA + TensorRT"]
  robot --> model["跃动模型 / 灵动模型"]
  model --> hw["人形本体（下载接口仅列 unitree 包）"]
  client --> rec["动作库录制 / mpds 数据采集"]
```

| 可获取项 | 实际内容 | 判定 |
|----------|----------|------|
| Windows 客户端 | 国内版 / 海外版安装程序，各约 829 MB | 闭源二进制 |
| 机器人端安装包 | `robot_type: unitree`；Ubuntu 20.04 + CUDA 11 + TRT 8、Ubuntu 22.04 + CUDA 12 + TRT 10 / 8；自解压 `.run`（约 1.27 GB），root 执行部署脚本；国内包标 `cn-beta.8` | 闭源二进制 |
| Jetson 环境检测脚本 | 读取 JetPack / CUDA / TensorRT 版本，给出应选安装包名 | 辅助脚本 |
| [GAEApp-API](https://github.com/westlakerobotics/GAEApp-API) | 客户端 WebSocket / HTTP 接口文档与 Python 示例（60 Hz 推送骨骼 / 关节帧），无 LICENSE | 对接接口，非算法代码 |
| 产品手册 | 飞书中英文手册：动捕设备配置、控制模式、动作库、数据采集 | 文档 |

- 产品端「跃动模型（不支持灵巧手）/ 灵动模型（更低延迟，支持灵巧手）」与论文中**单一执行器 + 延迟条件预判**的对应关系，官方未说明；推测为同一技术线的不同部署配置，不能把产品版本当作论文模型的复现。
- 机器人端包按 TensorRT 版本分发，推测执行器在 Jetson 机载以 TensorRT 推理；论文所述约 852M 参数规模是否原样部署，无法从安装包外层核实。
- 下载接口只列 `unitree` 机型包；官方宣称适配多种人形本体（自报），O1 等自研本体的部署包未在公开下载页中出现。

## 工程实践

复现时分别记录原始人体目标的噪声、生成器轨迹可行性、域随机化强度、端到端延迟分布、50 Hz 策略与 PD 环的时间戳。优先以零预判为基线，再扫 1–5 帧时域，报告**人与机器人同一绝对时间轴**上的跟踪误差及跌倒 / 任务成功率。

若只是想**使用**而非复现 GAE：可按官方下载页安装 V1.2.0 客户端与匹配 Jetson 环境的机器人端包，经动捕或 [GAEApp-API](https://github.com/westlakerobotics/GAEApp-API) 驱动；这属于商用产品集成（需加密狗授权），不能替代对论文训练流程的独立验证。

## 局限与风险

- 论文控制器主要依赖人体目标和机器人本体观测，**无 RGB / LiDAR 环境外感知**；楼梯与未知障碍无法由策略主动判断。
- 预判越远，跟踪误差越大；突发改向和不稳定通信仍需另外评测。
- 项目页展示物体操作，但灵巧手与物体状态的闭环能力不可由动作跟踪 SR 直接推出。
- 截至 2026-10-10，官方只提供闭源商用安装包与客户端 API，**无训练代码、模型权重、训练数据**；论文结果与部署细节仍不能独立复现，产品授权（加密狗 / 密钥）也限制了第三方评测。

## 源码运行时序图

**不适用**：截至 2026-10-10 官方只提供闭源商用二进制与客户端 API 示例，没有可辨识模块的训练 / 推理源码；不能把方法示意图或产品使用流程冒充源码模块时序。产品部署链路见上文「官方产品与开放状态」。

## 关联页面

- [遥操作](../tasks/teleoperation.md)
- [人形运动](../tasks/humanoid-locomotion.md)
- [运动重定向](../concepts/motion-retargeting.md)
- [Teleopit](./paper-teleopit.md)
- [TWIST2](./paper-twist2.md)
- [西湖机器人](./westlake-robotics.md)
- [TITAN O1](./westlake-titan-o1.md)

## 参考来源

- [论文原始资料](../../sources/papers/gae_general_action_expert_arxiv_2609_34233.md)
- [项目页开放状态核查](../../sources/sites/gae-general-action-expert.md)
- [西湖机器人官网 GAE 下载页与具身大模型模块（2026-10-10 核查）](../../sources/sites/wlrobo-gae-download.md)
- [westlakerobotics/GAEApp-API](https://github.com/westlakerobotics/GAEApp-API)

## 推荐继续阅读

- [原论文](https://arxiv.org/abs/2609.34233)
- [项目视频与方法概览](https://wangyf0928.github.io/gae-wlrobotics/)
- [官网 GAE 下载页](https://www.wlrobo.com/GAEDownload)
