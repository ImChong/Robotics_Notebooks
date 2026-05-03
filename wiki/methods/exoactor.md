---
type: method
tags: [video-generation, world-models, humanoid, motion-tracking, loco-manipulation, baai]
status: complete
updated: 2026-05-03
related:
  - ./generative-world-models.md
  - ../concepts/video-as-simulation.md
  - ./motion-retargeting-gmr.md
  - ../concepts/motion-retargeting.md
  - ../tasks/loco-manipulation.md
  - ../entities/unitree-g1.md
sources:
  - ../../sources/papers/exoactor.md
summary: "ExoActor 是 BAAI 提出的视频生成驱动的人形控制框架：把第三人称视频生成当作交互动力学的统一接口，通过 embodiment transfer + 动作链分解 + Kling 3 视频生成，再用 GENMO/WiLoR 做动作估计、SONIC 做通用动作跟踪，最终把 Unitree G1 在零真实数据下泛化到多难度交互任务。"
---

# ExoActor (视频生成驱动的交互式人形控制)

**ExoActor** 把"第三人称（exocentric）视频生成"作为人形机器人 **交互动力学的统一接口**：给定任务指令与场景观测，先让大型视频生成模型"想象"出一段任务执行视频，再把视频翻译为可执行的人形动作。它是 [Generative World Models](./generative-world-models.md) 与 [Video-as-Simulation](../concepts/video-as-simulation.md) 在人形控制层的一次具体落地。

## 核心思想：让视频生成模型当 demo 源

传统人形控制要么依赖显式仿真器，要么依赖大量真实采集的示教数据。ExoActor 把这两条路线绕开：

> **任务的"想象过程"由视频生成模型给出，机器人只负责把视频里的人体动作物理化执行。**

这种解耦带来三个特性：
- **零任务级真实数据**：只需要任务指令和一张第三人称初始观测，就可以零样本泛化到新场景。
- **天然支持长时序、富交互**：视频天然能表达走过去 → 蹲下 → 抓取 → 站起这种连贯过程。
- **模块独立可演进**：视频生成、动作估计、动作跟踪三段都可以独立升级。

## 主要技术路线

| 模块 | 关键技术 | 选型与说明 |
|------|---------|-----------|
| **Robot-to-Human Embodiment Transfer** | 提示驱动的图像编辑 | Nano Banana Pro / Gemini 3.1 Pro，把 G1 替换为姿态/朝向严格保持的人体 |
| **任务到动作链分解** | LLM 推理 | GPT-5.4 Thinking 输出原子动作链 $C=\{a_1,\ldots,a_T\}$ |
| **第三人称视频生成** | 大型视频生成模型 | 主用 Kling 3，相比 Veo 3.1 / Wan 2.6 物理合理性与 prompt 跟随更稳 |
| **全身动作估计** | 扩散式约束生成 | GENMO 输出 SMPL $(q_t, p_t)$ |
| **双手姿态估计** | 单图像手部姿态网络 | WiLoR + {open/half-open/closed} 三态 |
| **通用动作跟踪** | 大规模 motion tracking 控制器 | SONIC 作为"物理过滤器"，免除任务级 reward 工程；不接 GMR 等中间重定向 |
| **手部下发** | 关节级映射 | Unitree Dex3-1 兼容 7-DoF 关节 + event queue 同步 |

## 三阶段流水线

```
任务指令 + 第三人称初始观测
        │
        ▼
1. Video Generation
   ├─ Robot-to-Human Embodiment Transfer (Nano Banana Pro / Gemini 3.1 Pro)
   ├─ Task-to-Action 分解 + 结构化 Prompt (GPT-5.4 Thinking)
   └─ 第三人称定相机视频生成 (Kling 3 主用)
        │
        ▼
2. Interaction-aware Motion Estimation
   ├─ 全身动作: GENMO → SMPL 参数 (q_t, p_t)
   └─ 双手姿态: WiLoR + {open, half-open, closed} 三态
        │
        ▼
3. General Motion Tracking Deployment
   ├─ SONIC 大规模 motion tracking 控制器作为"物理过滤器"
   └─ 手部映射到 Unitree Dex3-1 兼容 7-DoF 关节
        │
        ▼
Unitree G1 物理执行
```

### Stage 1：第三人称视频-动作生成

直接让视频生成模型"画机器人"会因人形机器人外观与人类视觉先验不匹配而出现严重幻觉。ExoActor 用三个手段稳住这一阶段：

1. **Robot-to-Human Embodiment Transfer**：用基于提示的图像编辑（Nano Banana Pro / Gemini 3.1 Pro），在严格保持场景布局、相机视角、姿态、朝向、尺度、机器人本体比例的前提下，把图像中的 G1 替换成对应姿态的人。这一步把输入投影回视频生成模型擅长的"人类先验域"。
2. **任务到动作链分解**：用 GPT-5.4 Thinking 把高层指令 $G$ 拆成原子化、可视化、物理可执行的动作链 $C=\{a_1,a_2,\ldots,a_T\}$，例如 *Pick up the brown box and stand up* → *approach → bend down → grasp → lift → stand upright*。
3. **结构化 Prompt 模板**：把"动作链 + 初始观测"组织成 Shot / Scene / Motion / Execution / End State 五段式 prompt，固定相机、约束场景几何、强化"机器人风格"的运动模式。
4. **模型选型**：实测 Veo 3.1 / Kling 3 / Wan 2.6 后，**Kling 3** 在物理合理性、动作稳定性、prompt 跟随上整体最优，被选为默认。

### Stage 2：交互感知的动作估计

视频生成的产物是像素，机器人需要的是结构化运动。这一阶段同时关注全身和双手：

- **全身动作（GENMO）**：扩散式约束生成模型，以视频特征 + 2D 关键点为条件，输出时序一致、物理合理的 SMPL 参数序列 $\mathcal{M}=\{q_t, p_t\}_{t=1}^T$，并对部分遮挡帧做时序填补。
- **双手动作（WiLoR）**：逐帧估计双手 3D 姿态 $\mathcal{H}=\{h_t^l, h_t^r\}$，并把每只手映射为 {open, half-open, closed} 三态 $\mathcal{S}$。视点（正面 / 背面）会决定左右手的语义对应方式，避免手性歧义。

最终拼成联合表示：

$$ \tilde{\mathcal{M}}=\{q_t,\,p_t,\,h_t^l,\,h_t^r,\,s_t^l,\,s_t^r\}_{t=1}^{T} $$

这一步关键是**保持交互语义**——估计的轨迹中已经隐含了视频里的接触、空间约束和任务相关的运动模式，而不是单纯的关节角时间序列。

### Stage 3：通用动作跟踪部署

估计到的是高质量的"运动学参考"，但还缺少接触力、扭矩限制、平衡约束等动力学一致性。ExoActor 直接复用 [SONIC (Luo et al., 2025)](./beyondmimic.md) 这种规模化 motion tracking 控制器：

- **物理过滤器 (physics-filter) 视角**：让 SONIC 把"想象出来"的人体运动映射到 G1 的可行控制空间，即使是视频里出现的 parkour 风格大动作也能保持稳定。
- **不需要重定向**：作者有意省略 SMPLX → robot 的中间 retargeting 阶段（详见下文消融）。
- **手部部署**：把估计的双手三态映射到 Unitree Dex3-1 兼容的 7-DoF 关节目标，与 SMPL 主体轨迹通过 event queue 同步下发到机器人侧控制器。

## 任务难度分级与系统能力

ExoActor 按"导航/交互"复杂度把零样本任务分成三级：

| 难度 | 代表任务 | 主要考察 |
|------|---------|---------|
| **B (Easy)** | 走向桌上的瓶子 / 篮子；绕开椅子 | 稳定 locomotion + 空间一致的目标到达 |
| **A (Moderate)** | 把瓶子扫到一旁 / 走到椅子坐下 / 抬起箱子站起 / 钻过或跨过障碍 / 擦桌面 | 全身协调 + 粗粒度交互 |
| **S (Challenging)** | 多步操作：拾取物体放入篮子或垃圾桶；瓶子直立放进篮子 | 精细手物协调 + 空间精度 |

在 S 级任务上，由于动作估计在手部高度上仍有残余误差，作者会在目标物下垫支撑底座作为工程权宜方案，相关局限在论文 §4 单独讨论。

## 关键消融与失败模式

### 视频生成模型选型
Kling 3 vs Veo 3.1 vs Wan 2.6：Kling 3 在物理合理性、动作稳定性、prompt 跟随上整体更优；Veo / Wan 更易出现运动漂移、人-物交互不一致、突发性错误生成、末态不稳定。

### 重定向消融（与传统直觉相反）
对估计出来的 SMPLX 轨迹接 [GMR](./motion-retargeting-gmr.md) 或 OmniRetarget 等 [Motion Retargeting](../concepts/motion-retargeting.md) 模块：
- **能带来什么**：更平滑的全身运动，更少的高频抖动。
- **同时引入了什么**：明显的空间偏差。原因是估计动作本身就有全局位置漂移和脚滑，重定向尝试"修正"这些伪影时反而会破坏整体轨迹；同时人机体型/肢长差异会导致步长和位置积累误差。
- **结论**：在该流水线下，**直接把人体动作喂给 SONIC 比经过重定向再喂更稳**。这给"重定向永远是收益项"的传统假设提供了一个有意思的反例。

### 失败模式溯源
- **视频生成端**：物体幻觉（小伞被画成大伞）、动作序列错误、不真实的环境配置、末态姿势物理不合理。
- **动作估计端**：手腕方向错误（视频里在垂直抓瓶子，估计成水平腕）；遮挡和后视角时全身姿态恢复不完整；快速动作下精度下降。
- **执行端**：手部高度误差和步距误差导致够不到目标，需要工程补偿。

## 在本知识库中的定位

ExoActor 是一篇典型的"桥接型"论文，把好几条看似独立的技术线串在一起：

- 把 [Video-as-Simulation](../concepts/video-as-simulation.md) / [Generative World Models](./generative-world-models.md) 的"视频生成当世界模型"思想，从视觉策略训练扩展到**物理人形机器人控制**。
- 把 [Motion Retargeting](../concepts/motion-retargeting.md) 与 [GMR](./motion-retargeting-gmr.md) 放进消融，提示"重定向不是免费的"。
- 把 [BeyondMimic](./beyondmimic.md) / SONIC 风格的 motion tracking 大模型作为通用执行层，而不再为每个任务单独 reward 工程。
- 在 [Unitree G1](../entities/unitree-g1.md) 上端到端验证，是当前 G1 平台上"video → motion → tracking"系统的代表实现。

## 参考来源

- [sources/papers/exoactor.md](../../sources/papers/exoactor.md) — 本仓库 ingest 档案。
- Zhou Y., Ma J., Peng Y., Sun Z., Bai Y., Karlsson B. F. *ExoActor: Exocentric Video Generation as Generalizable Interactive Humanoid Control.* arXiv:2604.27711, BAAI, 2026. <https://arxiv.org/abs/2604.27711> / <https://arxiv.org/html/2604.27711v1>
- 项目主页：<https://baai-agents.github.io/ExoActor/>
- 相关基础工作：GENMO (Li et al., 2025)、WiLoR (Potamias et al., 2025)、SONIC (Luo et al., 2025)、Kling 3 (Kling Team, 2025)、SMPL (Loper et al., 2015)。

## 关联页面

- [Generative World Models](./generative-world-models.md) — ExoActor 把视频生成模型当成隐式世界模型来用。
- [Video-as-Simulation](../concepts/video-as-simulation.md) — 视频即仿真范式在人形控制上的具体实例化。
- [GMR (通用动作重定向)](./motion-retargeting-gmr.md) — ExoActor 的消融提供了"什么时候不该用重定向"的反例。
- [Motion Retargeting](../concepts/motion-retargeting.md) — 流水线中"是否需要中间重定向"的决策点。
- [BeyondMimic](./beyondmimic.md) — SONIC 是 BeyondMimic 路线的规模化延伸，被 ExoActor 直接当通用执行器。
- [Loco-Manipulation](../tasks/loco-manipulation.md) — ExoActor 在 G1 上验证的 A/S 级任务大部分属于 loco-manipulation。
- [Unitree G1](../entities/unitree-g1.md) — 端到端验证平台。
