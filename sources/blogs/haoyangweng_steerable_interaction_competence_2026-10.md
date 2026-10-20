# Beyond Motion Tracking: Steerable Physical Interaction Competence

> 来源归档（blog / 作者研究札记）

- **作者：** Haoyang Weng
- **类型：** blog / research note
- **英文原文：** <https://haoyangweng.pages.dev/blog/2026-10-steerable-interaction-competence/en/>
- **中文原文：** <https://haoyangweng.pages.dev/blog/2026-10-steerable-interaction-competence/zh/>
- **引用页：** <https://haoyangweng.pages.dev/blog/2026-10-steerable-interaction-competence/>
- **发表日期：** 2026-10
- **抓取方式：** 作者官网英文页与中文页直读；两页内容结构对应
- **一句话说明：** 主张把低层 S0 从自由空间运动跟踪器推进为可感知场景、在物理交互闭环中学习动力学、并通过丰富意图接口接受上层 S1 精细引导的交互控制器。

## 核心摘录（归纳，非全文）

### 主张与边界

- **空间 grounding 不等于动力学 grounding。** 更快地产生控制信号不会自动带来接触、负载与扰动下的闭环响应。文中将动力学 grounding 定义为策略在仿真或真机执行动作、观察其对身体/接触/物体的影响，并据反馈调整后续动作。
- **交互能力不应只看动作跟踪误差。** 关键是 S0 能否看见场景，并在接触、负载或环境变化时及时调整。运动学参考更准确或 S1 推理更强，不能替代 S0 缺少的物体动力学交互经验。
- **目标是可复用且可 steer 的交互 S0。** 文章区分自由空间 motion-centric tracker、带物体输入但任务专用的控制器，以及通过共享交互接口服务多任务/物体/接触模式的目标形态；后者是研究愿景，不是已验证的通用系统。

### 表征与实验说明

- **Scene awareness：** 建议使用真实机器人可获得的 RGB、深度或点云等场景输入，减少对预设物体坐标框架的依赖。HDMI 中以机器人坐标系下的物体位姿表示适用于单物体场景，但扩展到任意物体/场景会受限。
- **Interaction intent：** 不止传身体运动参考，还应描述机器人、物体和接触意图；可用机器人坐标系下的物体点未来运动，扩展到多物体时可考虑 flow 表示。接触位置、施力大小及柔顺/释放等意图不能从相同身体轨迹中可靠推断。
- **示范实验：** 作者称使用 HDMI 代码库和从 WEAVE 重定向的人–物交互示范，在仿真中与箱子、桌椅等物体交互。文章没有给出量化成功率或对照实验，因此应视为研究方向/概念论证，不能据此断言通用真机能力。

## 相关资料与开源状态

- HDMI 项目页与训练仓已在本库核查为公开：[`sources/sites/hdmi-project.md`](../sites/hdmi-project.md)、[`sources/repos/hdmi.md`](../repos/hdmi.md)；文章链接的 `EGalahad/hdmi` 代码地址与本库记录的 `LeCAR-Lab/HDMI` 仓库不同，需注意可能是 fork/镜像，本文未核实两者关系。
- WEAVE 代码、项目页和数据集已有归档：[`sources/repos/weave.md`](../repos/weave.md)、[`sources/sites/weave-xiaohu-art.md`](../sites/weave-xiaohu-art.md)、[`sources/papers/weave_arxiv_2609_16683.md`](../papers/weave_arxiv_2609_16683.md)。原文直接指向 WEAVE 数据集。
- 作者札记未提供独立代码、模型权重或数据集入口；复现实验需以其引用的 HDMI/WEAVE 资源为准，不能把札记本身当作开放实现。

## 对 wiki 的映射

- [VLA 与低级关节控制器（MPC/WBC）融合架构](../../wiki/queries/vla-with-low-level-controller.md) — 补充 S0 的场景感知、动力学闭环与意图接口视角。

## 参考链接

- [HDMI 项目页](https://hdmi-humanoid.github.io/)
- [HDMI codebase（作者文中链接）](https://github.com/EGalahad/hdmi)
- [WEAVE 数据集](https://huggingface.co/datasets/appolyn/Weave)
- [WEAVE 项目页](https://xiaohu-art.github.io/Weave/)
- [SONIC](https://nvlabs.github.io/SONIC/)
- [π0.7：A Steerable Model with Emergent Capabilities](https://www.pi.website/blog/pi07)
