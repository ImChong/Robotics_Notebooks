# Arcadia: Toward a Full-Lifecycle Framework for Embodied Lifelong Learning（arXiv:2512.00076）

> 来源归档（ingest）

- **标题：** Arcadia: Toward a Full-Lifecycle Framework for Embodied Lifelong Learning
- **类型：** paper / embodied-lifelong-learning / real2sim2real / vln / vla / humanoid
- **来源：** arXiv abs / PDF / HTML（v1）
- **原始链接：**
  - <https://arxiv.org/abs/2512.00076>
  - PDF：<https://arxiv.org/pdf/2512.00076>
  - HTML：<https://arxiv.org/html/2512.00076>
- **作者：** Minghe Gao\*、Juncheng Li（通讯）、Yuze Lin\*、Xuqi Liu\*、Jiaming Ji\*、Xiaoran Pan、Zihan Xu、Xian Li、Mingjie Li、Wei Ji、Rong Wei、Rui Tang、Qizhou Wang、Kai Shen、Jun Xiao、Qi Wu、Siliang Tang、Yueting Zhuang（\*同等贡献；部分工作完成于宇树实习）
- **机构：** 浙江大学（Zhejiang University）；宇树科技（Unitree）；北京大学（Peking University）；南京大学（Nanjing University）；群核科技（Manycore Tech）；字节跳动 Seed（ByteDance Seed）；阿德莱德大学（University of Adelaide）
- **入库日期：** 2026-08-16
- **一句话说明：** 把具身学习写成不可拆的四段闭环——自主采集、生成式重建、共享 VLN/VLA 骨干、Sim-from-Real 反馈；仿真导航/操作相对基线约 +7.07% / +11.08%，G1+Dex-3 真机 46/100 导航、27/100 操作；EmbodiedKit **部分开源**（训练/数据脚本可辨识，权重与完整闭环未发布）。

## 开源状态（核查 2026-08-16）

- **项目入口：** 论文列出 [Embodied-Arcadia/EmbodiedKit](https://github.com/Embodied-Arcadia/EmbodiedKit)（无独立 `*.github.io` 项目页；homepage 为空）。
- **部分开源：** 仓为公开 Python 仓（约 29★，无 License，2025-11-19 后无新 push）。根 README 仅 TODO（环境/Quickstart/重构未写）。子目录有可辨识入口：
  - `vln_data_generate/` — Isaac Sim 5.0 NavMesh + G1 录制
  - `vla_data_generate/` — Franka Lula RRT → RLDS
  - `vln_train/` — Qwen2.5-VL LoRA / SFT / DPO / GRPO（通用 VLM 微调 fork）
  - `vla_train/` — OpenQwenVLA 预训练 / 微调 / LIBERO 评测（文档要求 H20、≥80 GB）
  - `scene_replace/` — InternUtopia 资产替换
- **未随仓发布：** 论文中的 frontier 探索 + Isaac ROS/Nvblox、Gaussian-splat → USD 重建、Sim-from-Real 三通道反馈、权重与数据集。`robot_navigator.py` 仍硬编码本机 G1 USD 路径。
- **互指：** [`sources/repos/embodiedkit.md`](../repos/embodiedkit.md)；升格 [`wiki/entities/paper-arcadia.md`](../../wiki/entities/paper-arcadia.md)。未建 `sources/sites/`（无独立项目页）。

## 摘要级要点

- **主张：** 具身学习是 **生命周期问题**，不是单阶段优化。只做采集 / 仿真 / 学习 / 部署中的一环，无法持续改进。四段 **不可分解**：去掉任一段就退回一次性训练。
- **四段：** (1) 部署环境内自主探索采集；(2) 生成式场景重建与增强；(3) 共享多模态骨干同时服务 VLN 与 VLA；(4) 把真机轨迹拆成任务/场景/机器人反馈，写回仿真。
- **对照：** GRUtopia 扩仿真覆盖，NaVILA 把执行拉到真机，但都没有「部署经验 → 可编辑仿真资产 + 再训练」的持久回路。
- **四个缺口：** 外源数据依赖、预渲染静态场景、导航/操作架构碎片化、真机反馈稀疏（只有成败）。
- **仿真：** 导航平均约 **+7.07%**、操作约 **+11.08%**（相对对应基线）。VLN-CE-Isaac 上 Arcadia w/ feedback SR **50.1%**（NaVILA 45.1%；Tuning 44.9%）。
- **真机：** Unitree G1 + Dex-3；100 导航 + 100 操作。Arcadia **46 / 27**，NaVILA 导航 13、OpenVLA 操作 9。多目标/多物体组合约 **17%**。导航零样本；操作对桌面四物块微调双臂模型。
- **消融：** 静态训练集、检索式场景、稀疏反馈都会掉点；联合训练掉点最小，说明 VLN/VLA 可共享 VLM 骨干。

## 核心论文摘录（MVP）

### 1) 任务：具身终身学习是闭环，不是单点算法

- **链接：** <https://arxiv.org/abs/2512.00076> §1
- **摘录要点：** 外源 YouTube / 四足数据与目标人形部署错位；Matterport/Habitat 预渲染场景不可把部署变化写回；导航当 OBB、操作当固定相机桌面，监督无法跨任务流动；部署只记成败则长程错误无法定位。作者把「持续把真机经验写回资产与策略」称为 embodied lifelong learning。
- **对 wiki 的映射：**
  - [Arcadia](../../wiki/entities/paper-arcadia.md)
  - [数据飞轮](../../wiki/concepts/data-flywheel.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 2) 方法：自主采集 → 生成式 USD → 共享骨干 → Sim-from-Real

- **链接：** §3.1–3.4；Fig. 2–3
- **摘录要点：** Isaac ROS + Nvblox + frontier 信息增益探索，输出 RGB-D / LiDAR / IMU / 里程计。SpatialLM 出场景图，Gaussian splat 合成 USD，导入 Isaac Sim。共享 Qwen2.5-VL：导航用 A\* 的 7 原语离散控制，操作用 RRT 轨迹，格式对齐 VLN-CE / BridgeData V2。反馈 \(F_t^T=\lambda_1 R_t+\lambda_2\|s_{t+1}-s_t\|+\lambda_3\mathcal{L}_{\mathrm{conf}}+\lambda_4\mathcal{L}_{\mathrm{goal}}\)，外加场景扰动与硬件约束 \(F^R\)。
- **对 wiki 的映射：**
  - [Arcadia](../../wiki/entities/paper-arcadia.md)
  - [VLN](../../wiki/tasks/vision-language-navigation.md)
  - [VLA](../../wiki/methods/vla.md)
  - [NaVILA](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)

### 3) 评测：仿真增益主要来自数据对齐与闭环，不是更大模型

- **链接：** §4.1–4.4；Table 1–2
- **摘录要点：** 同架构同预算下，仅换自主采集数据（w/o feedback）已比 NaVILA 平均 SR **+2.7 pp**；加反馈后再涨。LIBERO 消融：骨干 76.5 → Arcadia **87.2**；静态集 72.9、检索场景 81.4、稀疏反馈 85.3。真机 46%/27% 仍低，组合指令更差。
- **对 wiki 的映射：**
  - [Arcadia](../../wiki/entities/paper-arcadia.md)
  - [OpenVLA](../../wiki/entities/openvla.md)
  - [操作](../../wiki/tasks/manipulation.md)

### 4) 开源边界：接口脚本 ≠ 全生命周期可复现

- **链接：** §6；GitHub README（2026-08-16 核查）
- **摘录要点：** 作者说主要动机是统一接口、让研究者少做工程。公开仓给出 Isaac 数据生成与 Qwen 系训练/评测脚本，但探索、3DGS 重建、反馈写回与权重未发布；根 README 仍是占位。复现只能跑子模块，不能当完整闭环栈。
- **对 wiki 的映射：**
  - [EmbodiedKit](../repos/embodiedkit.md)
  - [VLN 四范式开源复现](../../wiki/overview/vln-open-source-repro-paradigms.md)

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-arcadia.md`](../../wiki/entities/paper-arcadia.md) — 主实体页
- [`wiki/tasks/vision-language-navigation.md`](../../wiki/tasks/vision-language-navigation.md) — 生命周期 / 共享 VLN–VLA 分支
- [`wiki/methods/vla.md`](../../wiki/methods/vla.md) — 与导航共享骨干 + 真机反馈
- [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md) — Real2Sim2Real 闭环实例
- [`wiki/concepts/data-flywheel.md`](../../wiki/concepts/data-flywheel.md) — 部署写回资产/策略
- [`wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md`](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md) — 主要 VLN 基线
- [`wiki/entities/openvla.md`](../../wiki/entities/openvla.md) — 主要 VLA 基线
- [`wiki/overview/vln-open-source-repro-paradigms.md`](../../wiki/overview/vln-open-source-repro-paradigms.md) — 不可当新手可跑通栈
- [`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md) — LIBERO / BridgeData / G1 操作
- [`wiki/entities/awesome-real2sim2real.md`](../../wiki/entities/awesome-real2sim2real.md) — R2S2R 文献坐标

## 推荐继续阅读

- Gao, Li et al., *Arcadia: Toward a Full-Lifecycle Framework for Embodied Lifelong Learning* — [arXiv:2512.00076](https://arxiv.org/abs/2512.00076)
- [EmbodiedKit](https://github.com/Embodied-Arcadia/EmbodiedKit)
- Cheng et al., *NaVILA* — [arXiv:2412.04453](https://arxiv.org/abs/2412.04453)
- Kim et al., *OpenVLA* — [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
