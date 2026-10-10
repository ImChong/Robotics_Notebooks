# 光轮 SimReady 官方博文合集（3 篇 + 产品页）

> 来源归档（blog / 厂商官方博文）

- **标题：** Lightwheel SimReady 相关官方博文三篇（Newton 运动资产合作 / USD Search 资产检索 / SimReady 物理数据基础设施）+ SimReady Library 产品页
- **类型：** blog（厂商官方博客，lightwheel.ai/blogs）+ site（产品页）
- **来源：** 光轮智能（Lightwheel）
- **博客列表：** https://lightwheel.ai/blogs（日期以列表页为准；SimReady 主文正文页无日期，列表页标 Mar 16, 2026）
- **产品页：** https://lightwheel.ai/asset-library（SimReady Library）；资产商城 https://simready.com/
- **入库日期：** 2026-10-10（正文、产品页与开源状态均于 2026-10-10 抓取核对；正文为服务端渲染 HTML，simready.com 为前端渲染，用 Playwright 抓取）
- **一句话说明：** 光轮 SimReady 从「OpenUSD 资产库 + Newton 颗粒地形资产 + USD Search 检索」（2025-09）到「Measure → Solve → Generate 物理数据基础设施 + 加入 Newton TSC」（2026-03）的演进；资产数、性能数字均为 **光轮自报**。
- **沉淀到 wiki：** [`wiki/entities/lightwheel-simready.md`](../../wiki/entities/lightwheel-simready.md)；开源子集见 [`wiki/entities/cn-os-lightwheel-simready-asset.md`](../../wiki/entities/cn-os-lightwheel-simready-asset.md)
- **后续产品：** SimReadyGen（2026-07-20）见 [`lightwheel_simreadygen.md`](./lightwheel_simreadygen.md)

---

## 篇目清单

| 日期 | 标题 | URL |
|------|------|-----|
| 2025-09-29 | Lightwheel-Newton Partnership on Development of High-Quality Locomotion Assets for the Newton Physics Engine（作者 Zhen Liu, Yuting Jiang） | https://lightwheel.ai/media/lightwheel-newton |
| 2025-09-29 | Creating a simulation environment for robot training is hard, but accelerating asset discovery using USD Search makes it easier.（作者 Mustafa） | https://lightwheel.ai/media/lightwheel-usd-blog-CoRL |
| 2025-10-23 | SimReady Announces Special Pricing for Startups Including NVIDIA Inception Members（补充，列表页 + Playwright 抓正文） | https://lightwheel.ai/media/simready-announces-special-pricing |
| 2026-03-16 | SimReady: The Physics Data Infrastructure for Physical AI | https://lightwheel.ai/media/simready |
| （常驻） | SimReady Library 产品页 | https://lightwheel.ai/asset-library |

## 开源状态核查（2026-10-10）

| 项 | 结论 |
|----|------|
| SimReady 商业资产库（simready.com） | **商业为主 + 部分免费**：首页快照 24 件资产中 9 件标 "Open Sourced"（CC BY-NC 4.0，USD zip），其余按件售卖（样例价 **$50 / $145**）；NVIDIA Inception 成员 **15% 折扣**；产品页称"所有商业授权资产包含在 Lightwheel Lab Enterprise Package" |
| [LightwheelAI/Lightwheel-simready-asset](https://github.com/LightwheelAI/Lightwheel-simready-asset) | **已开源（非商用）**：README 称 **259** 个 USD 资产（251 操作 + 8 运动地形），目标 Isaac Sim 4.5 / 5，许可 **CC BY-NC 4.0**（根目录 LICENSE 存在），资产经 Google Drive 下载；要求命名 `lightwheel_{asset_name}` 署名 |
| [LightwheelAI/Newton-Lightwheel](https://github.com/LightwheelAI/Newton-Lightwheel) | **已公开**：USD 场景、Houdini HIP/HDA、`mpm_test.py` 加载脚本，约 **1.5M** 粒子；依赖 fork [LightwheelAI/newton_corl](https://github.com/LightwheelAI/newton_corl)；**根目录未见 LICENSE 文件**（LICENSE / LICENSE.md / LICENSE.txt 均 404），许可不明 |
| [LightwheelAI/Lightwheel-YCB](https://github.com/LightwheelAI/Lightwheel-YCB) | **已开源**：125 件 YCB SimReady 资产（106 原物体全量重建 + 19 积木块），MJCF + USD 双格式 |
| Physical Measurement Factory 测量数据 / 标定参数 | **未公开**（文内无数据集链接） |
| Newton 中 USD 可变形资产解析、SimReady schema | 文称与 NVIDIA 共同推进；**未列具体 PR / schema 仓库链接**（自报） |
| Hugging Face `LightwheelAI` | 13 个 dataset（EgoPro/EgoStandard/EgoDemo、Lightwheel-Tasks-*、leisaac-*、iros2026-ikea-assembly 等），**无 SimReady 资产库镜像** |
| GitHub API | 会话内 `api.github.com` 返回 403，仓库存在性以 `git ls-remote` + raw README 核对 |

## 1. 2025-09-29 · Lightwheel–Newton 运动资产合作

- 定位：Newton 由 NVIDIA、Google DeepMind、Disney Research 发起；光轮自称以"核心贡献者"加入，提供高质量仿真资产以加速 Newton 采用。
- 自我定位（自报）："SimReady 资产市场领导者"；覆盖刚体 / 铰接 / 刚-铰混合 / 可变形 / 流体资产；称刚上线的 **simready.com** 为 "the largest open source library of SimReady assets for Isaac Sim"（措辞需与 simready.com 实际"付费为主 + 部分 CC BY-NC"对照阅读）。
- 同期将 **YCB** 升级为跨平台（Isaac Sim `.usd` + MuJoCo `.mjcf`），资产 teleop-ready、RL-ready。
- **求解器选择**：Newton 中的 **Implicit MPM**（源自 Daviet 等 SIGGRAPH 2016，原为 CPU 实现）——理由：大规模稳定；适合沙/土/雪等连续介质；作为 Newton 与其 MPM 求解器的压力测试用例。
- **资产管线**：在 OpenUSD 资产上扩展 Newton 专属属性；Newton 内的场景解析管线自动组装物理属性与碰撞。
  - **网格资产**：12 m × 12 m 地形 + 岩石、树桩等障碍；DCC（Houdini）自动配置碰撞与材质，在 Isaac Sim 中正确渲染。
  - **双求解器碰撞**：Implicit MPM 直接用原始高精三角网格；**MuJoCo Warp** 预做凸分解并缓存。
  - **粒子资产**：沙 / 土 / 雪粒子系统在 DCC 中制作并存为 OpenUSD；每类粒子单独配内聚力、摩擦、弹性等参数。
- **实验**：Newton 示例中的 **ANYmal** 四足 + 平地预训练策略在可变形起伏地形上明显走不稳（定性结论，未给数值）；后续计划换人形。
- **工程问题**：不同粒子组材料约束需与 Newton 团队协作保证 Newton Beta 稳定；粒子穿透碰撞网格问题"近乎完全解决"；当前用自定义 schema 写求解器专属属性，计划迁移到 **Newton USD schema**；Houdini USD 互通问题已解决。
- **性能（自报）**：Windows + RTX 4090，约 **1.5M** implicit MPM 粒子时 **8–12 FPS**；刚体与 MPM 各自时间步同步；预期 Linux + 更强 GPU 达 **≥20 FPS**。
- **开源**：资产与环境发布于 GitHub `LightwheelAI/Newton-Lightwheel`；后续计划在 Isaac Lab 中贡献更多 Newton 资产。
- 附：2025-10-08 OpenUSD Insiders 直播 "Closing the Sim2Real Gap with SimReady and AI"（NVIDIA Madison Huang、Akhil Docca，光轮 Steve Xie）。

## 2. 2025-09-29 · USD Search 资产检索

- 光轮在 **simready.com** 上把 NVIDIA **USD Search API** 部署到约 **2,000** 件"最高质量、最严格策展"的操作/运动资产上；其中含标准化后的 **Lightwheel-YCB**。
- lightwheel.ai 上用 USD Search 的**类型检索**在 Lightwheel-YCB 子集内过滤，服务可复现实验。
- 三种模态：自然语言（如 "squishy things"、"kitchen items under 500 grams"）、以图搜资产、类型分类。
- 技术底座（转述 NVIDIA 规格）：NVCLIP（CLIP 商用实现，ViT-H，ImageNet top-1 77.86%，7 亿图训练），TensorRT 推理，Ampere 及以上 GPU，Linux 部署。
- 自报效果：文本/图像查询亚秒级响应；**预期**每日处理 10,000 次查询；"用户参与度可测提升"未给数值。
- 注意：USD Search 需经 NVIDIA Omniverse 商业授权获取。

## 3. 2025-10-23 · 初创与 NVIDIA Inception 折扣（补充）

- simready.com 对机器人初创（含 NVIDIA Inception 成员）**15% 折扣**。
- 自报：**2,000+** 资产，可为操作、导航、运动任务定制；含刚体、铰接、可变形、液体四类；**七大类**：服装、制造、电子、医疗、食品饮料、住宅、仓储。
- 演示环境：家庭导航、杂乱仓库 AMR、传送带拣包、零件分拣、机柜线缆整理（均在 Isaac Sim / Isaac Lab 中）。
- 页脚声明：simready.com 由光轮拥有和运营，**非 NVIDIA**；为 Isaac Sim 第三方资产市场。

## 4. 2026-03-16 · SimReady: The Physics Data Infrastructure for Physical AI

- 问题：仿真物理参数（摩擦、形变、接触）多为估计而非测量，导致策略上真机失败；"benchmark 常在测仿真器而不是物理世界"。
- 提出 **Lightwheel SimReady System**：遵循 **OpenUSD 内容规范**，资产是几何 + 物理 + 语义的结构化、机器可读描述，可跨测量 / 仿真 / 真实部署复用。
- 三段闭环 **Measure → Solve → Generate**：
  - **Measure**：自建 **Physical Measurement Factory**，用精密仪器与可重复实验测摩擦系数、刚度与形变、接触动力学、关节约束与运动范围；Real-to-Sim 验证测试对比仿真与受控真实实验，持续校准。
  - **Solve**：需覆盖刚体与铰接机构、布料/线缆等可变形体、粒子与流体、多接触操作——传统刚体仿真难以准确刻画。
  - **Generate**：基于 OpenUSD 的标准化管线批量生产物理一致资产与环境；资产按 **content profile** 编写，作为中立输入跨多引擎复用；支撑大规模遥操作采数、上千环境 RL、可复现评测、系统化场景生成。
- **与 NVIDIA / Newton 合作**：OpenUSD 作为 SimReady 资产与 Newton 多物理求解器之间的共享场景/数据模型，同一资产可喂 Newton、Isaac Sim 与其它 OpenUSD 工作流；光轮贡献：共同定义 SimReady 资产标准、在 Newton 管线中实现 **USD 可变形资产解析**、扩展复杂操作求解能力。
- **工业伙伴（自报）**：与 **Samsung** 共同开发/标定 Newton，使其装配机器人在仿真中掌握复杂线缆操作；与 **Analog Devices（ADI）** 结合其传感技术做 sensor-aware 仿真，对齐仿真与真实工业传感器的力/力矩/接触信号，用于高精度插装。
- **加入 Newton TSC**：光轮"将"以 Technical Steering Committee 成员身份加入 Newton（Linux Foundation 项目），聚焦：复杂物理资产的 SimReady 标准与 schema、求解器改进与真实标定、Real-to-Sim 资产调参工具、为 Newton 生态提供参考资产/场景与技术反馈。
- 未给出资产数量、测量精度或 sim–real 误差的量化数字。

## 5. SimReady Library 产品页（lightwheel.ai/asset-library，2026-10-10 快照）

- "Ready-to-Use Asset Library"：在 simready.com 获取全部经准备与验证的 3D 资产；**所有资产均含商业授权**；"即插即用，无需额外设置"；所有商业授权资产包含在 **Lightwheel Lab Enterprise Package**。
- 定制资产示例（Get Custom Assets）：
  - 农业：草莓采摘（茎具弹性与**可断关节**）
  - 医疗：**可变形肝脏切割**（任意位置实时切割、形变与阻尼）
  - 家庭：交互式住宅（家具/门/家电，刚体 + 铰链）
  - 服务机器人精密插装：电源插头插入（插入力建模 + 线缆形变）、插头插入（柔性线缆）、线缆穿管
  - 食品工业：汉堡组装（多层软体抓取堆叠）

## 6. simready.com 首页快照（2026-10-10，Playwright 渲染）

- 标题 "Robot Simulation Assets for Isaac Sim — Works with Teleop, RL, and VLA for both manipulation and locomotion"。
- 首屏 24 件资产：9 件 "Open Sourced"（Apartment 419 MB、KitchenRoom 972 MB、StandMixer、Blender、Stove/Stovetop 等，CC BY-NC 4.0）；其余 "Buy Now"：Stove/Stovetop/Blender **$145**、UtensilSet/UtensilRack **$50**；均为 USD zip，可"Customize"。
- 页面不展示资产总数；"2,000+" 以博文自报为准。

## 对 wiki 的映射

- [wiki/entities/lightwheel-simready.md](../../wiki/entities/lightwheel-simready.md) — SimReady 资产体系主节点
- [wiki/entities/cn-os-lightwheel-simready-asset.md](../../wiki/entities/cn-os-lightwheel-simready-asset.md) — 259 件 CC BY-NC 开源子集
- [wiki/entities/cn-os-lightwheel-ycb.md](../../wiki/entities/cn-os-lightwheel-ycb.md) — Lightwheel-YCB
- [wiki/entities/newton-physics.md](../../wiki/entities/newton-physics.md) — Newton 资产标准 / TSC / Implicit MPM 用例
