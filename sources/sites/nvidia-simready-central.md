# NVIDIA SimReady Central（Hugging Face 资产库）

> 来源归档

- **标题：** Announcing SimReady Central: An Open Library of Simulation-Ready 3D Assets for Robotics and AI Factories
- **类型：** site（NVIDIA 官方资产库公告与 Hugging Face 入口）
- **来源：** NVIDIA / Hugging Face
- **链接：** https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories
- **资产搜索：** https://huggingface.co/spaces/nvidia/simready-central
- **NVIDIA 官方数据集：** https://huggingface.co/datasets/nvidia/simready-assets（2026-11-21 核查：2,919 个 3D Assets、518 GB、CC BY 4.0；详见数据集来源归档）
- **发布：** 2026-10-21
- **入库日期：** 2026-10-24
- **通读：** 已完成
- **本次复核：** 2026-11-21（NVIDIA 2026-10-21 原公告及 2026-11-19 更新的 SimReady 官方指南）
- **一句话说明：** NVIDIA 将 2,000 多个面向机器人仿真与 AI 工厂数字孪生的 SimReady OpenUSD 资产汇集到 Hugging Face；资产来自多个供应商数据集，可由开发者和 Agent 检索、组合成仿真场景。
- **代码：** [NVIDIA-Omniverse/simready-central](https://github.com/NVIDIA-Omniverse/simready-central)（资产提交流程与索引工具，Apache-2.0；不是资产本身的统一许可证）
- **沉淀到 wiki：** 是 → [`wiki/entities/nvidia-simready-central.md`](../../wiki/entities/nvidia-simready-central.md)

---

## 官方要点

- **规模与访问：** 官方公告发布时称库中有 **2,000 多个**资产；入口免费且无需申请访问。公告描述的是机器人训练环境和 AI 工厂数字孪生用途。
- **资产类型：** 包含机器人手臂、夹爪、仓储设备、货架、服务器机架和冷却设备等；资产以 OpenUSD 形式提供，并按适用的 SimReady profile 编入仿真所需属性。
- **机器可用信息：** 视资产及 profile 而定，可包含碰撞几何、质量、摩擦、关节结构与限位、语义标签、材质和传感器属性。它们使资产能用于支持相应属性的仿真工作流；不能据名称假设每个模型都含有全部字段。
- **多方供给：** 库汇集参与 OEM 与资产提供方在 Hugging Face 上托管的数据集；例如 Comau、FANUC、Franka Robotics、Robotiq、Universal Robots、Vertiv 和 NVIDIA Robotics 相关资产。NVIDIA 仓库是审查提交的入口，不表示全部模型均由 NVIDIA 建模或统一授权。
- **Agent 工作流：** USD Search 可通过关键词或自然语言发现资产；Agent 可将所选资产引用进 OpenUSD 场景、编写场景组装脚本并生成预览。官方示例面向 Isaac Sim 的仓储抓取场景和 AI 工厂布局。

## 开放与许可边界

- SimReady Central 的公开搜索/浏览入口可免费使用；具体资产通过 Hugging Face 上的源数据集分发。
- [NVIDIA-Omniverse/simready-central](https://github.com/NVIDIA-Omniverse/simready-central) 仓库的 NVIDIA 编写代码采用 Apache-2.0，仓库主要负责 OEM 资产包提交清单、校验与本地浏览流程。
- **Apache-2.0 不适用于所有资产。** 每个包记录各自来源、版本和许可；下载、再分发或商用前应查看对应 Hugging Face 数据集与资产包的 license 字段及条款。
- 提交资产需要 OEM onboarding 和仓库协作者权限；普通开发者可浏览、搜索和使用公开资产，不因此获得提交权限。

## 2026-11-21 增量核实

- **正确链接：** 官方数据集为 `nvidia/simready-assets`（不是 `nvidia/nsimready-assets`）；官方公开搜索 endpoint 为 https://nvidia-simready-central.hf.space/ 。
- **开发者加载方式：** NVIDIA [SimReady Central 官方文档](https://docs.omniverse.nvidia.com/simready/latest/simready-central.html)指出可直接在 Isaac Sim/Kit 中引用 Hugging Face CDN URL、使用 HF API 下载，或使用 Isaac Sim Asset Search 扩展；均需检查目标 profile、验证证据及运行时。
- **质量与检验：** [官方验收规则](https://github.com/NVIDIA-Omniverse/simready-central/blob/main/docs/acceptance-criteria.md)要求 profile 验证、benchmark 及物理运行测试；不应把“可检索”视作“已证明任意操作任务的物理精度”。
- **许可范围：** 官方文档注明所有审核通过的许可可进入 HF 搜索和 Central 浏览，但面向中国分发和 Isaac Sim 内置浏览器限标准宽松 CC BY 4.0 资产。不同供应商数据集独立授权。
- **ABB 99% 数据口径：** [ABB 2026-03-09 新闻稿](https://www.abb.com/global/en/news/134030/prsrl-abb-robotics-partners-with-nvidia-to-deliver-industrial-grade-physical-ai-at-scale)说的是 RobotStudio HyperReality 与 Omniverse 技术融合；与 10 月 SimReady Central 发布并非同一测评。
- **传播/瑕疵：** `nvidia/simready-assets` 当前月下载数可核验；“发布两天登顶趋势榜”“质量 3 吨货架”“关节驱动归零且官方承认修复”未取得可复核一手记录，不作为事实入库。

## 对 wiki 的映射

- NVIDIA 资产库与 Agent 工作流 → [`wiki/entities/nvidia-simready-central.md`](../../wiki/entities/nvidia-simready-central.md)
- NVIDIA Physical AI 数据集集合 → [`wiki/entities/nvidia-physical-ai-datasets.md`](../../wiki/entities/nvidia-physical-ai-datasets.md)
- Lightwheel 自有 SimReady 资产系统 → [`wiki/entities/lightwheel-simready.md`](../../wiki/entities/lightwheel-simready.md)

## 官方材料

- [NVIDIA/Hugging Face 公告](https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories)
- [SimReady Central Hugging Face 搜索入口](https://huggingface.co/spaces/nvidia/simready-central)
- [NVIDIA SimReady Assets 数据集](../datasets/nvidia-simready-assets.md)
- [NVIDIA-Omniverse/simready-central 仓库](https://github.com/NVIDIA-Omniverse/simready-central)
- [SimReady Foundation 规范与校验工具](https://github.com/NVIDIA/simready-foundation)
