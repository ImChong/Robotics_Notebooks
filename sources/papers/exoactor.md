# exoactor

> 来源归档（ingest）

- **标题：** ExoActor: Exocentric Video Generation as Generalizable Interactive Humanoid Control
- **类型：** paper
- **来源：** arXiv abs / arXiv HTML
- **原始链接：**
  - <https://arxiv.org/abs/2604.27711>
  - <https://arxiv.org/html/2604.27711v1>
- **项目主页：** <https://baai-agents.github.io/ExoActor/>
- **作者单位：** Beijing Academy of Artificial Intelligence (BAAI)
- **入库日期：** 2026-05-03
- **最后更新：** 2026-05-03
- **一句话说明：** ExoActor 把"第三人称视频生成"作为人形机器人交互行为建模的统一接口，再通过人体动作估计 + 通用动作跟踪控制器，把生成视频转化成 Unitree G1 的可执行行为，实现零真实数据的零样本任务泛化。

## 核心摘录

### 1) ExoActor: Exocentric Video Generation as Generalizable Interactive Humanoid Control（Zhou, Ma, Peng, Sun, Bai, Karlsson 等，BAAI，2026）
- **链接：** <https://arxiv.org/abs/2604.27711>
- **核心问题：** 现有人形控制策略难以泛化到新场景下"机器人 ↔ 环境 ↔ 物体"的交互富集（interaction-rich）行为；扩大任务分布通常依赖昂贵的真实数据采集和定向调参。
- **核心思路：** 把大规模预训练的 **第三人称（exocentric / 第三方视角）视频生成模型** 作为隐式世界模型，输出"想象出来的执行过程"作为高层 demonstration；再用一个可独立改进的模块化下游链路把"像素级想象"翻译为机器人的可执行动作。
- **三阶段流水线：**
  1. **Video Generation（视频生成）**：给定任务指令 + 第三人称初始观测，先做 **Robot-to-Human Embodiment Transfer**（用 Nano Banana Pro / Gemini 3.1 Pro 做提示驱动的图像编辑，把 Unitree G1 替换成姿态/朝向/尺度严格保持的人体），再用 GPT-5.4 Thinking 把高层指令拆解为原子化动作链 $C=\{a_1,\ldots,a_T\}$，构建包含 Shot/Scene/Motion/Execution/End State 的结构化 prompt 模板，最终主要使用 **Kling 3** 生成第三人称、固定相机、动作连贯的视频。
  2. **Interaction-aware Motion Estimation（交互感知动作估计）**：对生成视频用 **GENMO**（Li et al., 2025，扩散式约束生成）恢复 SMPL 参数下的全身 3D 运动 $\mathcal{M}=\{q_t,p_t\}$；再用 **WiLoR**（Potamias et al., 2025）逐帧估计双手姿态 $\mathcal{H}=\{h_t^l,h_t^r\}$，并把每只手映射到 {open, half-open, closed} 三态 $\mathcal{S}$；最终得到联合表示 $\tilde{\mathcal{M}}=\{q_t,p_t,h_t^l,h_t^r,s_t^l,s_t^r\}$。
  3. **General Motion Tracking Deployment（通用动作跟踪部署）**：直接把估计的人体运动喂给 **SONIC**（Luo et al., 2025）这种大规模动作跟踪基础模型，让其充当"物理过滤器"，在不做任务级 reward 工程、不做机器人专属重定向的前提下，让 G1 完成稳定执行；手部映射到 Dex3-1 兼容的 7-DoF 关节，通过 event queue 与 SMPL 主体轨迹同步下发。
- **关键设计点：**
  - **embodiment transfer**：在视频生成之前先把机器人替换成人，避免预训练视频模型在 robot 视觉先验上幻觉；同时也提升后续 SMPL/手部估计的稳定性。
  - **不需要重定向**：作者实验表明，叠加 GMR / OmniRetarget 等 SMPLX→robot 的中间重定向反而引入空间偏差与轨迹漂移，因此最终系统选择"人体动作 → SONIC 直接吃"。
  - **任务难度分级评测**：B 级（基本导航/避障）、A 级（粗交互，如扫物、坐下、跨越障碍）、S 级（多步精细操作，如把瓶子直立放进篮子）。
  - **失败模式溯源**：视频生成阶段的物体幻觉、动作估计阶段的手腕方向错误（垂直抓取被估成水平腕部）、执行阶段的手部高度误差（论文中通过在目标物下垫高补偿）。
- **消融与选型：**
  - **视频生成模型对比：** Veo 3.1 / Kling 3 / Wan 2.6，Kling 3 在物理合理性、动作稳定性、prompt 跟随上整体最优，被选为默认。
  - **重定向消融：** 引入 GMR/OmniRetarget 能让全身动作更平滑、抖动更少，但会放大全局位置漂移与脚滑；最终去掉重定向。
- **意义：**
  - 把"视频生成模型 + 通用动作跟踪控制器"组合成一种新的人形控制范式，把交互建模的负担从控制层下移到生成模型层。
  - 是一篇非常清晰地连接 **Generative World Models / Video-as-Simulation / Motion Retargeting / Motion Tracking / Loco-Manipulation** 的桥接型工作，可作为本知识库讨论"视频生成驱动的人形控制"的代表方法页。
- **对 wiki 的映射：**
  - 新建 [ExoActor (视频生成驱动的交互式人形控制)](../../wiki/methods/exoactor.md)
  - 在 [Generative World Models](../../wiki/methods/generative-world-models.md) 与 [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) 中补充 ExoActor 作为"以视频生成模型当 demo 源"的代表
  - 在 [GMR (通用动作重定向)](../../wiki/methods/motion-retargeting-gmr.md) 与 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) 中补充"重定向并非永远收益为正"的反例论据
  - 在 [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) / [Whole-Body Coordination](../../wiki/concepts/whole-body-coordination.md) 中关联其难度分级评测视角
  - 在 [Unitree G1](../../wiki/entities/unitree-g1.md) 中作为 G1 平台上"video → motion → tracking"系统的范例

## 当前提炼状态

- [x] arXiv abs 与 HTML v1 原文核心方法与实验信息已摘录
- [x] wiki 页面映射确认
- [x] 相关 wiki 页面的参考来源段落已补 ingest 链接
- [ ] 若后续视频生成驱动控制方向资料增多，可拆出 video-to-humanoid pipeline / generative demo source 子页面
