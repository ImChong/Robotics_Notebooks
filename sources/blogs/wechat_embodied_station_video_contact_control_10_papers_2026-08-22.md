# 具身智能又卷到哪了？10 篇开源论文把视频、接触和控制串起来

> 来源归档（blog / 微信公众号）

- **标题：** 具身智能又卷到哪了？10 篇开源论文把视频、接触和控制串起来
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/EmC4gNgcQdPX34vxy-qSVQ
- **发表日期：** 2026-08-22
- **入库日期：** 2026-08-22
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md`](../raw/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md)
- **一句话说明：** 汇总 10 篇近期具身/机器人论文（文内均给项目页或代码链），主线从「看懂人类视频」到「生成可执行动作」，再到「跨技能持续适配」；**10/10 均有独立 `paper-*` 详情节点**（本 ingest **新建 9**、**复用 1 既有 complete**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：这批工作共同追问如何把人类视频、仿真结构、接触几何与大模型表征转化为真机可执行能力。**DreamHand** 与潜动作研究把人类视频变成可学习动作表征；**Video2DoorTraversal**、**AdaPT** 与 **ROS 2 Panda 栈**把仿真、跟踪与控制接口推向部署；**CoToGrasp**、**GOAG** 与 **PVRA** 则强调接触拓扑、装配依赖与物体无关建模仍是操作泛化关键。

### 10 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | DreamHand | [2608.20308](https://arxiv.org/abs/2608.20308) | **待发布**：GitHub 仓已建，README 写明推理/权重/训练 **未发布** | [paper-dreamhand](../../wiki/entities/paper-dreamhand.md) |
| 02 | Video2DoorTraversal | [2608.20251](https://arxiv.org/abs/2608.20251) | **待发布**：项目页 **Code Coming soon** | [paper-video2door-traversal](../../wiki/entities/paper-video2door-traversal.md) |
| 03 | RoMAN-Flow | [2608.20208](https://arxiv.org/abs/2608.20208) | **已开源** 训练/评测 + HF 权重 | [paper-roman-flow](../../wiki/entities/paper-roman-flow.md) |
| 04 | AdaPT（人形网球） | [2608.20087](https://arxiv.org/abs/2608.20087) | **部分开源**（复用既有节点） | [paper-adapt](../../wiki/entities/paper-adapt.md) |
| 05 | PVRA | [2608.19968](https://arxiv.org/abs/2608.19968) | **已开源** 训练/推理/评测 | [paper-pvra](../../wiki/entities/paper-pvra.md) |
| 06 | CoToGrasp | [2608.19776](https://arxiv.org/abs/2608.19776) | **未开源**：项目页无 GitHub 链 | [paper-cotograsp](../../wiki/entities/paper-cotograsp.md) |
| 07 | GOAG | [2608.19759](https://arxiv.org/abs/2608.19759) | **未开源**：项目页无 GitHub 链 | [paper-goag](../../wiki/entities/paper-goag.md) |
| 08 | FER ROS 2 Panda 栈 | [2608.19740](https://arxiv.org/abs/2608.19740) | **待发布**：站点有演示；论文链 **匿名 open-science** | [paper-fer-ros2-panda-stack](../../wiki/entities/paper-fer-ros2-panda-stack.md) |
| 09 | What Matters for Latent Actions | [2608.19613](https://arxiv.org/abs/2608.19613) | **已开源** GitHub + HF 模型/数据 | [paper-latent-actions-matter](../../wiki/entities/paper-latent-actions-matter.md) |
| 10 | OrthoSkillVLA | [2608.19589](https://arxiv.org/abs/2608.19589) | **已开源** 训练/仿真评测 | [paper-orthoskillvla](../../wiki/entities/paper-orthoskillvla.md) |

### 文内要点速记

1. **DreamHand** — 视频扩散模型作确定性几何编码器；遮挡/出画双手轨迹恢复；ARCTIC/HOT3D MPJPE-p ↓30%/40%。
2. **Video2DoorTraversal** — 单段 RGB 视频 → DoorTwin 仿真门孪生 → ArticuACT 双深度闭环；五扇真门 96.57%，未见门 zero-shot 80.95%。
3. **RoMAN-Flow** — AR-NF 离线 RL + sampling-free advantage-weighted likelihood + 一步蒸馏；仿真/真机 competitive 且降推理延迟。
4. **AdaPT** — 转播/MoCap 学职业网球风格；速度自适应规划–跟踪；G1 与 Atom 真机。
5. **PVRA** — RGB-D 3D 关键点投票学装配依赖；渐进装配 pose 与 Step SLA-AUC。
6. **CoToGrasp** — 接触拓扑条件灵巧抓取；canonical workspace 物体无关训练；DexGraspNet SOTA。
7. **GOAG** — 只学夹爪接触流形；推理时接入物体特征；MultiDex 86.93%。
8. **FER ROS 2** — 异步硬件接口 + rate-matching + 位置域参考；两平台 Panda 规划/柔顺/遥操作验证。
9. **Latent Actions** — 41 项 LAM 设计统一实证；LAPO/ΔDINO 强基线；VLM+潜动作微调更强初始化。
10. **OrthoSkillVLA** — VLM/ActionHead 分组件正交子空间 + 轻量 MoE decoder；缓解 VLA 连续学技能遗忘。

## 对 wiki 的映射

- **10/10 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 9** 个实体；**AdaPT** 在当日先前 ingest 已有 complete 页 → **只回链博客，不重复造页**。
- 阅读坐标：[视频–接触–控制 10 篇技术地图](../../wiki/overview/video-contact-control-10-papers-technology-map.md)（**非**论文详情替代，仅作横切面索引）。
- 交叉：[Egocentric Vision](../../wiki/tasks/egocentric-vision.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Dexterous Grasping](../../wiki/tasks/dexterous-grasping.md)、[VLA](../../wiki/methods/vla.md)、[Offline RL](../../wiki/methods/offline-rl.md)、[AdaPT](../../wiki/entities/paper-adapt.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 10 篇独立节点核查（9 新建 / 1 复用 / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
