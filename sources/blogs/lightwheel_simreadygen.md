# 光轮 SimReadyGen 发布博文

> 来源归档（blog / 厂商官方博文）

- **标题：** Introducing SimReadyGen — Agentic Simulation Generation for Physical AI
- **类型：** blog（厂商官方博客）+ site（Web 应用）
- **来源：** 光轮智能（Lightwheel）
- **链接：** https://lightwheel.ai/media/simreadygen
- **产品入口：** https://lightwheel.ai/simreadygen/ （试用 https://lightwheel.ai/simreadygen/try）
- **发布日期：** 2026-07-20（博客列表页；正文页无日期）
- **入库日期：** 2026-10-10
- **一句话说明：** 光轮的 agentic 仿真资产生成引擎：文本 prompt（可选参考图）→ 结构化 OpenUSD SimReady 资产，物理参数来自 SimReady Foundry 实测管线；集成 NVIDIA Omniverse Content Agents；以积分制 Web 服务提供，**无论文、无代码**。
- **前置产品：** [SimReady 博文合集](./lightwheel_simready.md)
- **沉淀到 wiki：** [`wiki/entities/lightwheel-simreadygen.md`](../../wiki/entities/lightwheel-simreadygen.md)

---

## 开源 / 论文核查（2026-10-10）

| 项 | 结论 |
|----|------|
| 论文 / arXiv | **未找到**：博文无论文链接；arXiv API 检索 `SimReadyGen`、`Lightwheel AND SimReady` 均 0 结果 |
| SimReadyGen 源码 | **未开源**：博文与 Web 应用无 GitHub 链接；`LightwheelAI/SimReadyGen` 的 `git ls-remote` 要求认证（不存在或私有） |
| 生成资产 / 数据集 | 无公开 HF 数据集（`LightwheelAI` HF 组织 13 个 dataset 中无 SimReadyGen 相关） |
| 依赖的 NVIDIA 组件 | **已开源**：博文直链 [nvidia-omniverse/content-agents](https://github.com/nvidia-omniverse/content-agents)（README 标题 "USD Content Agents"，亦可经 `NVIDIA-Omniverse/usd-content-agents` 访问；Apache 2.0；Geometry / Material / Texture / Physics / Joint / Validation 六类 agent，Material、Physics 为 Beta，其余 Research Preview；自述为参考实现，非独立 text-to-3D 生成器） |
| 访问方式 | **商业 Web 服务**：需登录（含 Google 登录），积分钱包经 **Stripe** 充值或兑换码；具体价格未在公开页面显示 |

## 官方要点摘录（博文）

- **定位**：机器人训练需要规模与速度都超过传统资产制作的物理准确环境；SimReadyGen 是光轮的 **agentic simulation-generation engine**，**基于 OpenUSD**、集成 **NVIDIA Omniverse Libraries**，从文本 prompt 生成结构化、仿真就绪资产。
- **Measured Physics, Generated at Scale**：背后是 **SimReady Foundry**（实测物理管线）——起点是 **Physics Measurement Factory**，测量真实物体的接触、摩擦与动力学，作为仿真 ground-truth 物理参数；同一测量数据也驱动光轮物理求解器开发。由此形成"大型且持续增长"的实测 SimReady 资产库；SimReadyGen 在 Omniverse Libraries 之上、以此为基础生成，**"每个资产都携带实测物理，而非估计"（自报，未说明生成物体如何映射到实测参数）**。
- **Built for OpenUSD Workflows**：集成 NVIDIA Omniverse Content Agents，支持 USD 文件的材质分配、物理属性分类、纹理生成、内容校验；产物可进入 Isaac Sim / Isaac Lab 等工作流。
- **持续学习闭环**：生成资产与环境 → **RoboFinals**（工业级评测，在实测物理下测策略）→ **RoboStack**（端到端部署管线）→ 真实表现数据回流优化下一轮仿真；"Generate, evaluate, deploy, and learn"，SimReadyGen 是每轮起点。
- 页面示例：Bucket Hat、Refrigerator、Teddy Bear、Oven（文本 → 参考图 → SimReady 资产预览）；例："A cream-white bucket hat with a rounded crown, wide brim, stitched band, and two metal eyelets."
- 未给出生成速度、成功率、物理误差、资产数量等任何量化数字。

## Web 应用前端观察（2026-10-10，从公开 JS 包的界面字符串归纳 → 推测，非官方文档）

- 生成流程界面文案：输入安全检查 → "Clarification Agent"（查歧义、缺失尺寸、风险）→ 判定物体需 **刚体 / 关节 / 可变形 / 静态** → 生成 2+ 张参考图（"通常 20–60 秒"）→ 选择参考图后生成（铰接资产**必须**先选参考图）→ "Mapping component structure and collider hints"、"Splitting visible parts, rails, handles, panels" → 最终 SimReady 资产 → **交互式 Isaac Sim 预览**（流式）→ 下载 / 邮件交付。
- prompt 上限 4000 字符；参考图 ≤12 MB；输入框示例聚焦"单件家具或道具"。
- 另有 **CAD 平面图**（PNG/JPG，≤15 MB）上传入口与 `submitSceneJob`，提示上传"清晰单层平面图（墙、房间标签、门窗）"——推测为**场景级生成**入口。
- 前端进度估计常量：rigid / deformable 各阶段合计约 30 分钟、articulated 约 70 分钟（推测为预估进度条参数，非 SLA）。
- 计费：积分（credits）钱包、Stripe 结账、兑换码、发票；"Production / Exhibition display" 模式用于展会演示。

## 对 wiki 的映射

- [wiki/entities/lightwheel-simreadygen.md](../../wiki/entities/lightwheel-simreadygen.md) — 主节点
- [wiki/entities/lightwheel-simready.md](../../wiki/entities/lightwheel-simready.md) — 上游实测资产体系
- [wiki/entities/lightwheel-robofinals.md](../../wiki/entities/lightwheel-robofinals.md) — 闭环下游评测
