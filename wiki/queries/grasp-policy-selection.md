---
type: query
tags: [manipulation, grasping, perception, vla, il, rl, 6dof, anygrasp, graspnet]
status: complete
updated: 2026-05-21
summary: "抓取策略选型 Query：从「开放场景 vs 已知物体」「稀疏 vs 稠密候选」「几何启发式 vs 学习方法 vs 端到端策略」三轴出发，给出真机抓取系统的方案选择与组合路径。"
related:
  - ../methods/grasp-pose-estimation.md
  - ../entities/anygrasp.md
  - ../tasks/manipulation.md
  - ../methods/contact-net.md
  - ../methods/visual-servoing.md
  - ../concepts/contact-rich-manipulation.md
  - ../comparisons/anygrasp-vs-graspnet.md
sources:
  - ../../sources/papers/perception.md
  - ../../sources/repos/anygrasp-sdk.md
  - ../../sources/papers/imitation_learning.md
---

> **Query 产物**：本页由以下问题触发：「真机抓取系统里，到底该用几何启发式、检测式 grasp pose，还是端到端策略？开放场景和已知物体的选型逻辑有什么不同？」
> 综合来源：[Grasp Pose Estimation](../methods/grasp-pose-estimation.md)、[AnyGrasp](../entities/anygrasp.md)、[Manipulation](../tasks/manipulation.md)、[ContactNet](../methods/contact-net.md)、[Visual Servoing](../methods/visual-servoing.md)、[Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)

# Query：抓取策略选型（开放场景 vs 已知物体 / 稀疏 vs 稠密 / 几何 vs 学习）

## TL;DR 决策树

```text
你是否已知具体物体（CAD / Mesh / 已扫描点云）？
├── 已知 (单一物体 / 固定 SKU 集合)
│   ├── 工作站标定稳、视角可控
│   │   └→ 6D 位姿估计 + 离线规划好的 grasp set
│   └── 视角/姿态多变
│       └→ 类别级 6D + 模板匹配 → 检测式 grasp 兜底
└── 未知 / 开放词汇 (bin picking / 服务机器人)
    ├── 场景结构清晰（桌面 / 框内 / 货架）
    │   ├── 平行夹爪 + 静态场景
    │   │   └→ 检测式 grasp（GraspNet / Contact-GraspNet / AnyGrasp）
    │   └── 动态 / 移动物体 / 边走边抓
    │       └→ 检测式 grasp + 跨帧关联（AnyGrasp 时序跟踪）
    └── 任务高度依赖语义 / 语言指令
        ├── 任务可分解为「找到目标物 → 标准 grasp」
        │   └→ 开放词汇分割 → 检测式 grasp
        └── 任务包含「用法 / 工具操作 / 接触序列」
            └→ VLA / IL 端到端策略（DP / ACT / π₀ 等），grasp 退化为内隐技能
```

## 快速结论

- **不要把抓取看成单一问题**：先把场景按「物体已知度 / 候选稠密度 / 任务抽象度」三轴分类，再决定走哪条路线。
- **80% 工业 bin picking 场景的最佳起点仍是「检测式 grasp pose」**，而不是端到端策略 —— 工程可控、可加显式碰撞检查、可与现有 IK/规划器无缝拼。
- **稠密 ≠ 一定更好**：很多任务用 graspness 热图 + Top-K 比裸稠密候选更快、更稳；稠密只是给下游排序留更多余量。
- **VLA / IL 适合的不是「抓得更准」，而是「会用 / 会接触 / 会恢复」**：当抓取本身不是终点而是技能链的一环时，再让策略层吞掉这一段。
- **几何启发式从未被淘汰**：在已知物体 / 已知工作面的工业场景，离线 grasp set + 在线姿态对齐至今仍是最稳的方案。

## 三轴选型对比表

### 轴 1：物体已知度（开放场景 vs 已知物体）

| 场景 | 主线方案 | 关键数据/模型 | 失败模式 | 兜底 |
|------|---------|--------------|---------|------|
| 单 SKU 工业件 | 6D 位姿估计 + 离线 grasp set | CAD + AprilTag / 模板匹配 / FoundationPose 类位姿网络 | 反光/相似件、相机标定漂移 | 触觉/电流反馈纠偏 |
| 类别级（杯子、瓶子等） | 类别 6D / 关键点 → 模板抓取 | NOCS、Cat-NOCS 系；或类内 grasp set | 类内形状/材质差异大 | 检测式 grasp 接力 |
| 开放词汇 / 未知物体 | 检测式 6-DoF/7-DoF grasp | [GraspNet-1Billion](../methods/grasp-pose-estimation.md)、[Contact-GraspNet](../methods/contact-net.md)、[AnyGrasp](../entities/anygrasp.md) | 透明/反光、薄片、长杆 | 触觉 + 重抓 / scripted recovery |
| 语言指令 + 未知场景 | 开放词汇分割（SAM/Grounded SAM/OWL-ViT）+ 检测式 grasp | 分割掩码 → 区域抓取 | 文本到掩码歧义、远距离误检 | 让 VLM 多轮澄清 / 用户确认 |

### 轴 2：候选稠密度（稀疏 vs 稠密）

| 路线 | 输出形态 | 优势 | 风险 | 典型代表 |
|------|---------|------|------|---------|
| 稀疏采样 + 打分 | 数十~数百候选 | 简单可控；与传统规划器易接 | 启发式漏检；候选偏置严重 | GPD、PointNetGPD |
| 稠密 6-DoF 回归 | 每个点位都给候选 | 覆盖率高、可全场景排序 | 显存与延迟成本；后端碰撞检查瓶颈 | GraspNet-1Billion |
| 稠密 + 接触点锚定 | 每接触点 1 个紧凑参数化 | 训练稳定、稠密但维度低 | 仍依赖局部几何质量 | [Contact-GraspNet](../methods/contact-net.md) |
| graspness 热图 + Top-K | 先筛后回归 | 工程化最友好，部署延迟可控 | 阈值/Top-K 调参需要 | GSNet / Graspness、[AnyGrasp](../entities/anygrasp.md) |
| 端到端动作（无显式 grasp） | 直接出动作 chunk | 与下游技能自然串联 | 可解释性弱，难做碰撞证明 | Diffusion Policy / ACT / VLA |

### 轴 3：方法类型（几何启发式 vs 学习方法 vs 端到端策略）

| 方法层 | 输入 | 输出 | 工程链路 | 何时优先 |
|--------|------|------|----------|---------|
| 几何启发式（antipodal / force-closure / 离线 grasp set） | 已知 mesh / 局部 patch | 解析候选 + 力学评分 | 完全白盒、可证 | 单 SKU、已知物体、低不确定性场景 |
| 检测式学习（GraspNet 系） | 点云 / RGBD | 6-DoF/7-DoF 候选 + 质量分 | 与规划器解耦，可逐段调试 | 开放词汇、bin picking、未知物体的最稳起点 |
| 时序 / 跟踪扩展 | 多帧点云 | 跨帧关联抓取链 | 兼容动态场景与移动机械臂 | 边走边抓、动态物体、连续伺服 |
| 端到端策略（IL / RL / VLA） | 像素 / 多模 | 动作 chunk | 黑盒；难做单点碰撞证明 | 抓取嵌在技能链中、需要复杂用法 / 接触 / 恢复 |

## 推荐组合 pipeline

下面四种是真机系统里出镜率最高的「检测 + 执行 + 兜底」组合，按工程化难度递增：

1. **已知物体 / 工业站**
   - 6D 位姿（AprilTag / 模板）→ 离线 grasp set → IK / [cuRobo](../entities/curobo.md) 规划 → 阻抗执行
   - 失败兜底：触觉滑移检测 → scripted regrasp
2. **桌面 / bin picking（开放词汇）**
   - 深度/点云 → [Grasp Pose Estimation](../methods/grasp-pose-estimation.md)（GraspNet 或 [AnyGrasp](../entities/anygrasp.md)）→ Top-K + 显式碰撞检查 → IK → 阻抗闭合
   - 最后几厘米切到 [Visual Servoing](../methods/visual-servoing.md) / 触觉对齐
3. **动态场景 / 边走边抓**
   - 多视点 / 移动相机 → AnyGrasp 类跨帧关联 → 抓取链平滑 → loco-manipulation 控制层
   - 必须保留 **位姿一致性检查**（同一物体跨帧 id 不漂移）
4. **任务级抓取（语言指令 / 工具使用）**
   - 任务可分解：开放词汇分割 → 检测式 grasp → 后续技能由 IL/VLA 接管
   - 任务不可分解：直接 [VLA](../methods/vla.md) / [Diffusion Policy](../methods/diffusion-policy.md) / ACT，抓取退化为内隐子技能

## 关键工程经验

### 1. 检测式 grasp pose 是「候选生成器」，不是「执行器」

网络打分高 ≠ 物理可执行：网络看不到机械臂运动学约束、看不到机座位姿、看不到工件附近的夹具。**任何检测式管线都应保留显式碰撞检查与可达性筛选**（常见落点是 [cuRobo](../entities/curobo.md) / MoveIt 的并行 IK + 碰撞过滤），把网络当成候选生成器。

### 2. AP 高 ≠ 真机成功率高

GraspNet AP / AP_novel 高，主要意味着「数据分布内匹配好」，但部署成功率受 **手眼标定 / 相机噪声 / 摩擦不确定性 / 抓取后晃动** 共同影响。真机调优时优先盯三个指标：

- 抓起后保持 > 5 s 的成功率
- 整 bin 清空所需的均次抓取（MPPH）
- 错误恢复成功率（首次抓失败后 N 次重试内是否能救回）

### 3. 透明 / 反光 / 薄片：从结构光开始就出错

深度相机本身在这些材质上深度缺失，**任何检测式 grasp 网络都救不回来**。常见救法：

- 双相机 / 偏振 / 主动结构光升级（硬件层）
- 单目深度估计兜底 + 谨慎使用
- 数据增强：用合成透明物体训练专用 head
- 退路：识别为「危险类」时切换到吸盘 / 不同末端

### 4. 端到端策略想替代抓取栈，要先认清三件事

很多团队尝试用 VLA / Diffusion Policy 直接吃掉「抓取」，最后才发现：

- **数据成本**：检测式管线一晚就能起 baseline，端到端需要数百~数千条 demo
- **可解释性差**：失败时只能整体重训，无法像检测式那样定位是「位姿错了」还是「夹爪闭合时刻错了」
- **碰撞证明缺失**：保险/工业场景仍会要求显式碰撞检查；端到端策略需要在外面再套一层 grasp set 安全校验

**结论**：端到端策略适合「抓取只是技能链中的一环」的任务（开柜门、倒水、插插头），而不是「让 AnyGrasp 退役」。

## 什么时候应该把抓取交给策略层

满足下列任意条件，再考虑用端到端策略吞掉抓取：

- 任务在抓起后立刻进入接触丰富阶段（插装、对齐、按压），grasp 不是终点
- 任务需要语义性「用法」选择（拿杯子要从把手抓，拿刀要从刀柄抓）
- 已经有足够多的人类演示数据，且演示覆盖了大部分失败恢复
- 工程上能容忍偶尔的"黑盒失败"且具备 scripted fallback

## 常见误区

- **「AnyGrasp 比 GraspNet 更新所以一定更好」** —— AnyGrasp 在 **动态 / 跟踪 / 工程化** 上更强，但权重不开源；做研究复现 / 白盒改造时 GraspNet baseline + Contact-GraspNet 仍是更合适的起点。
- **「6-DoF 抓取就能搞定灵巧手」** —— 本页谱系默认平行夹爪。多指 / 灵巧手抓取是另一套表征（接触面分配、力闭合 / form closure），见 [In-hand Reorientation](../methods/in-hand-reorientation.md)。
- **「先做端到端，效果不好再加 grasp 模块」** —— 顺序往往应反过来：先用检测式 grasp 把 baseline 跑通，再用 IL/VLA 替换其中可学的环节。
- **「只看 AP，不做物理试验」** —— AP 是离线匹配指标，真机部署前必须跑一轮仿真物理 + 真机试抓，确认抓后保持稳定。
- **「稠密候选 = 安全余量」** —— 没有显式碰撞检查的稠密候选只是 GPU 浪费；候选量越大，后端排序与可达性筛选的延迟越糟。

## 一句话记忆

> 默认起点是 **检测式 6-DoF grasp pose**；已知物体往左走「6D + 离线 grasp set」，技能链复杂往右走 **IL/VLA**；但「显式碰撞检查 + 触觉/视觉兜底」三件套永远不能省。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |
| DR | Domain Randomization | 训练时随机化仿真参数以提升跨域鲁棒迁移 |
| IK | Inverse Kinematics | 满足末端/姿态约束求解关节角的运动学逆解 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| IL | Imitation Learning | 从专家演示学习策略，奖励难定义时的主路线 |
| CAD | Computer-Aided Design | 计算机辅助设计，硬件结构建模 |
| DoF | Degrees of Freedom | 自由度，人形通常 20–50+ 关节 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |
| ACT | Action Chunking Transformer | 预测动作块的序列模型架构，常与 ALOHA 配套 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| GPU | Graphics Processing Unit | 图形处理器，大规模并行仿真训练的算力基础 |
| SDK | Software Development Kit | 软件开发工具包 |

## 参考来源

- [sources/papers/perception.md](../../sources/papers/perception.md) — 抓取感知与 6D 位姿估计来源整理
- [sources/repos/anygrasp-sdk.md](../../sources/repos/anygrasp-sdk.md) — AnyGrasp / GraspNet 工程化 SDK 与生态
- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md) — IL / VLA 把抓取作为内隐技能的路线
- Fang H. et al. (2020). *GraspNet-1Billion*. CVPR.
- Sundermeyer M. et al. (2021). *Contact-GraspNet*. ICRA. — <https://arxiv.org/abs/2103.14127>
- Fang H. et al. (2023). *AnyGrasp*. IEEE T-RO. — <https://arxiv.org/abs/2212.08333>

## 关联页面

- [Grasp Pose Estimation（抓取位姿估计）](../methods/grasp-pose-estimation.md) — 本 Query 的方法谱系基础页
- [AnyGrasp（抓取感知 SDK）](../entities/anygrasp.md) — GraspNet 系第三代工程化代表
- [ContactNet](../methods/contact-net.md) — 与 Contact-GraspNet 在「接触面预测」思路同源
- [Manipulation](../tasks/manipulation.md) — 操作任务总览，抓取是其感知子问题
- [Visual Servoing](../methods/visual-servoing.md) — 抓取最后几厘米的亚毫米级对齐方案
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md) — 抓握后的接触阶段执行层
- [cuRobo](../entities/curobo.md) — 抓取候选到无碰撞规划的下游求解器
- [Query：做机器人操作用模仿学习还是 RL？](./il-for-manipulation.md) — 抓取嵌在技能链时的策略层选型
- [Query：接触丰富操作实践指南](./contact-rich-manipulation-guide.md) — 抓握后的接触排错与调试顺序
- [AnyGrasp vs GraspNet：抓取检测家族选型对比](../comparisons/anygrasp-vs-graspnet.md) — 在检测式 grasp 路线内部，进一步在 SDK 与白盒基线之间做选型
