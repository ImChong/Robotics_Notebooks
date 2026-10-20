---
type: entity
project_id: viggle-jst-2
project: https://viggle.ai/
tags: [world-models, 3dgs, real2sim, character-animation, viggle]
topic: [world-models, sim2real]
status: complete
updated: 2026-10-21
related:
  - ./marble-world-model.md
  - ./atlas-world-model.md
  - ../methods/generative-world-models.md
  - ../concepts/functional-taxonomy-world-models.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/sites/viggle-jst-2.md
summary: "Viggle JST-2 是闭源的 Joint Space-and-Time 世界模型产品：用 Physics-Native Tokens 表示持续存在的 3D atoms，统一生成和重建场景、可动角色与全身动作；官方称单场景生成约 1 秒，但尚无公开权重、源码或数据，且场景交互仍使用规则物理。"
---

# Viggle JST-2（Joint Space-and-Time 世界模型）

**JST-2** 是 Viggle 于 2026 年发布的空间生成模型：它在共享的 3D 表示中处理场景、角色和动作，可从稀疏图像重建并扩展场景，也能从文本或视频生成角色动作。官方称其面向「物理原生」世界建模；这指模型表示包含空间位置与学习到的动态属性，不能直接理解为经过验证的刚体物理引擎或机器人仿真器。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| JST | Joint Space-and-Time | Viggle 用于统一空间与时间的模型架构名称 |
| 3DGS | 3D Gaussian Splatting | JST 解码器可输出的三维场景表示 |
| API | Application Programming Interface | 通过 Viggle 托管服务调用模型的接口 |
| MCP | Model Context Protocol | Viggle 提供的代理接入方式之一 |

## 为什么重要

- **场景、角色与动作共用一套 3D 表示：** 相比仅在像素视频中预测后续帧，JST-2 的 Physics-Native Tokens 带有显式空间位置，并为物体部分保留身份；官方宣称同一架构支持场景生成、重建、角色生成与动作生成。
- **生成速度值得关注：** 官方报告称在 NVIDIA B200 上场景推理约 1 秒，最多以 64 个视角逐步扩展空间上下文。该结果是厂商口径，且计时不含网络延迟，需与独立复现区分。
- **与机器人 Real-to-Sim 相关但不等同：** 稀疏观测生成场景可作为仿真资产探索方向，但外观完整不保证几何测量精度、碰撞正确性、物体动力学或接触可信度。

## 核心表示与数据流

JST 将世界表示为在空间中持续存在的 atoms。每个 atom 表示一块物质或物体部件，并由 Physics-Native Token 编码其 xyz 位置、属性向量和动态向量。官方示意的处理路径包括：

1. **编码观测：** 将一张或多张场景图片映射为 Physics-Native Tokens。
2. **从提示生成：** 也可由多模态提示直接生成 token 表示。
3. **条件解码：** 给定观察视角及运动或力输入，预测位置和属性的变化。
4. **输出：** 解码成图像或 3D Gaussian splats；可在浏览器中换视角继续查看，也可生成新视角扩展上下文。

## 能力与证据边界

| 能力 | 官方披露 | 评估时应注意 |
|------|----------|------------|
| 3D 场景生成 / 扩展 | 文本或图像生成场景；官方称支持最多 64 个视角上下文 | 官方报告 B200 单场景约 1 秒、推理成本低于 $0.01；不含网络延迟，尚无独立复现 |
| 稀疏观测重建 | 官方演示从真实场景的少量视角重建完整环境 | “视觉上忠实”是厂商陈述，不等同测绘精度或未知区域真实重建 |
| 角色生成 | 从文本或图像生成可动 3D 角色，官方称无需独立 rigging / skinning 步骤 | 角色基准为厂商设计的盲测；应留意测试集、基线版本和生成设置 |
| 文本 / 视频到动作 | 生成或重建全身动作；动作可应用于 JST-2 角色和标准骨架角色 | 动作质量、脚底接触和快速复杂动作仍有局限，详见官方动作页 |
| 场景中的交互动态 | token 含学习到的动态向量 | 官方说明 JST-2 的角色运动由学习动态驱动，但场景碰撞和物体交互仍依赖规则物理；JST-3 才计划替换这一部分 |

## 工程实践

| 项目 | 当前可核实信息 |
|------|----------------|
| 产品入口 | [PINOC / Viggle](https://viggle.ai/) |
| API | [Viggle 开发者页](https://viggle.ai/developers)；托管调用，不是本地推理入口 |
| 场景表示 | 官方介绍可输出 3D Gaussian splats；新视角可在本地浏览器查看 |
| 代码 / 权重 / 数据 | 截至 2026-10-21，官方项目页与 API 页未列出 JST-2 公开代码仓库、模型权重或训练数据 |
| 比较数据 | 角色与动作网页披露了与商业基线的盲测结果；结果属于 Viggle 自评，需按其测试协议解读 |

对机器人研究者，适合把 JST-2 视为**闭源的 3D 场景与动作生成服务候选**，先核对导出格式、相机和坐标约定、资产许可、场景碰撞网及 API 条款。若训练闭环策略，仍需独立验证碰撞、质量与惯量、摩擦、接触、可控物体状态和仿真步进接口；现有官方材料不足以证明 JST-2 本身提供这些能力。

## 局限与风险

- **闭源服务：** 截至入库日，未找到官方公开的训练 / 推理源码、权重或数据集；无法本地复现，也不能将 API 访问写成模型开源。
- **“物理原生”不代表完整物理仿真：** 官方说明场景碰撞和物体交互仍依赖规则物理，JST-3 才计划以学习动态替代。展示中的受力响应不应视为可用于机器人接触训练的动力学验证。
- **重建会受到生成先验影响：** 稀疏视角下未观测区域可能由模型补全；需要真实几何或可验证碰撞时，应与扫描 / 重建、解析仿真流程比较。
- **性能与基准为厂商披露：** 单场景速度、成本和基准胜率均需结合模型版本、输入、硬件、网络和评测协议理解。
- **截图中的“全球首个”是发布方定位语：** 本页记录官方口径，不将宣传用语当作独立验证结论。

## 关联页面

- [Marble（多模态 3D 世界生成产品）](./marble-world-model.md) — 生成式场景资产和可视化对照
- [Atlas（World Labs omni 世界模型）](./atlas-world-model.md) — 对照 3D spatial context 与世界生成路线
- [生成式世界模型](../methods/generative-world-models.md) — 生成世界的主要范式
- [世界模型功能分类](../concepts/functional-taxonomy-world-models.md) — 区分生成、重建与仿真能力
- [Sim2Real](../concepts/sim2real.md) — 评估生成场景迁移到机器人训练的约束

## 参考来源

- [Viggle JST-2 官方资料归档](../../sources/sites/viggle-jst-2.md)
- [JST-2 Character 官方评测](https://viggle.ai/research/jst2-character)
- [JST-2 Motion 官方评测](https://viggle.ai/research/jst2-motion)

## 推荐继续阅读

- [Viggle 产品与 JST-2 入口](https://viggle.ai/)
- [Viggle API / 开发者页](https://viggle.ai/developers)
- [JST-2 Motion: full-body 3D motion from text or video](https://viggle.ai/research/jst2-motion)
