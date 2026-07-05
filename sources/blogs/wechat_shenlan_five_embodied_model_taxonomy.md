# 五大具身模型详解：VLM、VLA、VLN、VLX、世界模型

> 来源归档（blog / 微信公众号）

- **标题：** 五大具身模型详解：VLM、VLA、VLN、VLX、世界模型
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/xj-rc6v64Ge6onoUPvkHLg
- **发表日期：** 2026-07-05
- **入库日期：** 2026-07-05
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；正文约 1.2 万字 / 20 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始落盘：** [wechat_shenlan_vlm_vla_vln_vlx_wm_2026-07-05.md](../raw/wechat_shenlan_vlm_vla_vln_vlx_wm_2026-07-05.md)
- **一句话说明：** 从统一 Transformer + 多模态编码底座出发，按「感知→导航→执行→融合→推演」递进拆解 VLM、VLN、VLA、VLX 与世界模型（WM）的输入输出、能力边界与协同链路；强调 VL 系列直接对接硬件执行，WM 专注时序虚拟预演。

## 核心摘录（归纳，非全文）

### 统一底层

- 五类模型共享 **神经网络拟合 + 多模态混合编码 + Transformer 主干**；差异来自模态配比、输出任务头、时序跨度与部署场景。
- **多模态编码**：图像、文本、深度/空间、本体姿态、时序变化 → 统一隐向量空间，支撑跨模态对齐与特征复用。
- **WM vs VL 系列**：WM 输出 **不参与即时硬件执行**，用于虚拟仿真推演与策略优化；VLM/VLN/VLA/VLX 输出对接真实感知、导航或动作。

### 五类模型递进关系

| 模型 | 定位 | 输入（文内归纳） | 输出 | 与上游关系 |
|------|------|------------------|------|------------|
| **VLM** | 跨模态感知理解底座 | RGB/帧序列 + 自然语言 | 语义描述、物体关系、指令解析（**无动作**） | 最前置；为 VLN/VLA/WM 供环境认知 |
| **VLN** | 空间导航专用 | VLM 双模态 + 深度/拓扑/障碍 | 全局路径、局部轨迹、避障参数 | VLM 延伸；**仅移动**，无力控/操作分支 |
| **VLA** | 端到端执行载体 | 视觉+语言+本体姿态+关节+空间 | 底盘轨迹、关节角、末端力控等 **可执行控制** | 整合 VLM+VLN，单主干拟合观测→动作 |
| **VLX** | 融合型通用架构 | 复用前三类全模态 | **并行**输出感知/导航/动作三类结果 | 单网多分支，替代多模型串联 |
| **WM** | 时序推演 | 环境观测 + 候选动作序列 | 未来多帧视觉/物理状态变化 | 独立于执行链；与 VLA 形成「决策+预演」 |

### VLA × 世界模型协同（文内主流范式）

- **VLA**：瞬时决策，基于当前观测生成可落地动作方案。
- **WM**：接收候选动作，在虚拟空间逐帧推演环境反馈与风险。
- 组合链路：**即时决策 → 虚拟预演 → 择优执行**，降低真机试错成本。

### 产业趋势（文末）

分立模型高精度落地、一体化通用模型（VLX 方向）、世界模型赋能优化三条主线并行；术语仍在快速融合。

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [五大具身模型分类（对比）](../../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) | **主沉淀页**：递进关系、I/O 边界、协同链路 |
| [VLA（方法）](../../wiki/methods/vla.md) | VLA 端到端执行定位与开源谱系 |
| [VLN（任务）](../../wiki/tasks/vision-language-navigation.md) | VLN 导航任务定义与评测 |
| [World Action Models](../../wiki/concepts/world-action-models.md) | 与文内 WM 的「联合预测」叙事对照 |
| [世界模型技术地图](../../wiki/overview/world-models-15-open-source-technology-map.md) | 开源 WM 生态补全 |
