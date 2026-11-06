# 伽马智能（Gamma Robotics）官网、GitHub 组织与成立报道核查

> 来源归档（site / Gamma Robotics 官方 + 媒体）

- **URL：** <https://gammarobotics.com.cn/>（Research <https://gammarobotics.com.cn/research>；产品 `/product/gammabot`、`/product/gamma-data`；开发者 `/developers`；新闻 `/news`）
- **GitHub 组织：** <https://github.com/Gamma-Robot>（显示名「Gamma Robotics（伽马智能）」）
- **机构：** 官网页脚「© 2026 GAMMA ROBOTICS / Gamma Evolution Intelligence Technology (Shanghai) Co., Ltd」；ICP 备案「沪ICP备2026056754号-1」。中文工商全称官网与报道均未写。
- **核查日期：** 2026-11-06
- **抓取方式：** 本环境直连官网 HTTPS 被连接重置，改用 WebFetch 读取官网首页、`/research`、`/product/gammabot`、`/product/gamma-data`、`/news`、`/developers`（均 HTTP 200）；`/careers` 未读。GitHub 组织页与 5 个仓库页用 WebFetch 读取，GitHub API 返回 403 未取得 `created_at`。arXiv abs / HTML 页逐篇读取版本历史与机构列表。
- **覆盖 wiki：** [伽马智能（Gamma Robotics）](../../wiki/entities/gamma-robotics.md)
- **通读：** 已完成

## 成立报道（媒体转述 TeleAI 官宣）

| 报道 | 日期 | 原文要点（摘录） |
|------|------|------------------|
| [IT之家（凤凰网转载）](https://tech.ifeng.com/c/8xOzOB5LWWf) | 2026-10-21 | 「中国电信人工智能研究院（TeleAI）10 月 20 日官宣，具身智能公司“伽马智能（Gamma Robtics）”近日在上海市徐汇区注册成立。」 |
| [上观新闻（中国青年网转载）](http://news.cyol.com/gb/keji/articles/2026-10/21/content_aj4N57ILEp.html) | 2026-10-21 | 「近日在上海市徐汇区注册成立，入驻模速空间」；「聚焦自进化人形机器人全模型创新及应用，并将与TeleAI共建人形机器人实验室」 |
| [网易智能（中华网转载）](https://tech.china.com/article/20261021/202610211967496.html) | 2026-10-21 | 「伽马智能（Gamma Robotics）」；「自进化主要指机器人通过真实环境交互持续积累经验、更新模型能力；全身模型则聚焦人形机器人全身运动与灵巧手操作的协同」 |
| [机器人大讲堂（搜狐）](https://www.sohu.com/a/1090545752_489960) | 2026-10-21 | 同上观新闻口径 |
| [新浪](https://k.sina.cn/article_5953189932_162d6782c067051jp4.html) | 2026-10-21 | 「该公司是由中国电信集团投资、TeleAI孵化的独立市场化主体」 |

- **注册日：** 各报道均写「近日」，未给具体日期；唯一带日期的是 TeleAI **2026-10-20 官宣**。TeleAI 原始公众号文章未找到链接。
- **英文名拼写：** 四家报道写「Gamma Robtics」，网易智能写「Gamma Robotics」；官网与 GitHub 均为 Gamma Robotics，判前者为通稿笔误（推测）。
- **人员：** 「白辰甲博士任创始人及首席执行官」，曾任 TeleAI 通用具身智能研究所所长；中国电信集团 CTO、首席科学家、TeleAI 院长李学龙「出任科学顾问」。其他成员未披露。
- **投资：** 「伽马智能由中电信（上海）人工智能产业私募基金进行战略投资」；该基金由中国电信与上海国投先导基金、央企战略性新兴产业发展基金、徐汇区上海汇资等共同出资。金额未披露。
- **合作：** 与 TeleAI 共建人形机器人实验室；「围绕中国电信 AIDC 算力基础设施服务、家庭和商业服务等方面打造示范应用」（上观新闻）。
- **技术方向（报道口径）：** 「自进化」与「全身模型」；「推动全身运动与灵巧操作的统一建模，使移动、平衡、抓取与操作衔接为完整的任务能力」。报道未提产品、论文或开源项目名称。

## 官网逐页记录

### 首页

- 标语：「EVOLVE WHAT MATTERS」；页脚「Gamma Robotics,towards physical world,evolve what matters」；另有「From Value to Action」。
- 使命（原文）：「Gamma Robotics is building general-purpose humanoid intelligence for the physical world, combining highly dynamic locomotion with precise whole-body manipulation so robots can move, adapt, and get useful work done in environments built for people.」
- 地址：「Model-Speed Space Innovation Community Xuhui District, Shanghai」（即模速空间）；电话 +86 180 1719 9312。
- 研究支柱只有标题 Locomotion / Manipulation / Self-Improvement，无说明文字；首页无任何数据规模或部署数字。
- 机构 logo（未说明是合作方、投资方还是成员来源）：上海交大、复旦、清华、Noitom、Lightwheel、Flexiv、GENTO、Unitree、Manycore、Kinetix AI、TeleAI。
- 社媒：微信二维码；X / Instagram / LinkedIn / 小红书均标「Account coming soon」。

### 产品：GBot（`/product/gammabot`）

- 定位：人机交互 + 具身操作 + 自主导航，「so robots can understand needs and continuously complete tasks in real spaces」。
- 操作模型（自报）：「Our PRTS manipulation foundation model is pretrained on 167B operation and embodied-reasoning data.」「Deployed dual-arm tasks reach 95.9% average success, with no individual task below 90% and long-horizon tasks reaching 95%.」任务清单与评测条件未给。
- **GBot-W 规格（官网表格）：**

| 项 | 数值 |
|----|------|
| 尺寸 | 670 × 550 × 1340–1800 mm（高度可调） |
| 重量 | 140 kg |
| 自由度 | 22（不含两端执行器与移动底盘）：颈 2、单臂 7、腰 3、腿 3 |
| 单臂负载 | 额定 6 kg（正文另写「12 kg rated dual-arm payload」） |
| 重复定位精度 | ±0.03 mm |
| 底盘 | 四舵轮，最高 1.5 m/s，转弯直径 780 mm，最大坡度 10° |
| 算力 | NVIDIA Jetson AGX Orin，275 TOPS，64 GB |
| 电池 / 续航 | 34.4 Ah（LiFePO₄ 17.2 Ah ×2），48 V；混合工况 4–8 h |
| 传感 | 12 MP RGB ×2、双目（Gemini 335 等）、LIVOX MID-360 ×3、超声 ×8 |
| 末端 | 可选灵巧手、夹爪等 |

- 价格、发布日期、量产状态：官网未写。

### 产品：GData（`/product/gamma-data`）

- 定位：「High-precision collaborative data capture」，是数采硬件而非机器人。
- **GData-Finger：** 手持仿生柔顺夹爪，双目深度、IMU、高精度编码器，160° 超广角双目、60 mm 基线，2 h 热插拔电池。
- **GData-Ego：** 头戴四相机，「A 260° effective field of view and 180° depth coverage」，9 轴 IMU 500 Hz。
- 共同指标（自报）：多设备无线同步 <1 ms，末端空间定位 1 mm。

### 开发者（`/developers`）

- 三层栈原文：System 2「Reason with value」— 任务意图、长时程目标与进度感知决策；System 1「Plan whole-body behavior」— 把决策转为人体尺度机器人的受约束运动；System 0「Execute on the embodiment」— 边缘侧高频控制、跟踪与适配。
- 开发闭环：DEFINE → SIMULATE → CONNECT → DEPLOY → LEARN。
- 开放状态：架构说明与 Research「AVAILABLE」；「SDK & INTERFACES」标「STAGED ACCESS」（「Model, simulation and deployment interfaces are opening in stages.」）；「EMBODIMENTS」标「IN DEVELOPMENT」。**页面无 SDK、API、文档或下载链接**，也无模型名、频率数值。

### 新闻（`/news`）

- 只有「Updates are on the way.」，无条目。

### Research（`/research`）

官网列 12 项，按页面分组；链接多数指向 **TeleHuman** GitHub 组织或项目页，不在 Gamma-Robot 组织下。

| 分组 | 官网日期 / 状态 | 工作 | 链接 |
|------|-----------------|------|------|
| Locomotion | 2026-02，CoRL 2026 | TextOp | <https://github.com/TeleHuman/TextOp> |
| Locomotion | 2026-02，RSS 2026 | HUSKY 人形滑板 | <https://github.com/TeleHuman/humanoid_skateboarding> |
| Locomotion | 2026-09，Under Review | NEXUS: A Perceptive Foundation Policy for Cross-Domain Whole-Body Teleoperation | <https://nexus-humanoid.github.io/> |
| Manipulation | 2026-04，CoRL 2026 | PRTS | <https://github.com/TeleHuman/PRTS> |
| Manipulation | 2025-09，ICLR 2026 | Align-Then-stEer | <https://github.com/TeleHuman/Align-Then-Steer> |
| Manipulation | 2026-04，CoRL 2026 | AssemLM | <https://github.com/TeleHuman/AssemLM> |
| Loco-Manipulation | 2026-10，Under Review | KODA | <https://koda-humanoid.github.io/koda/> |
| Loco-Manipulation | 2026-10，Under Review | Test-Time Compute for Humanoid BFM | <https://bfmttc.github.io/> |
| Simulation | 2026-10，Technical Report | SMART | <https://teamillusion-smart.github.io/> |
| Self-Improvement | 2026-10，CoRL 2026 | GR-BC | <https://gr-bc.github.io/> |
| Self-Improvement | 2026-08，Under Review | WNM-3D | <https://github.com/TeleHuman/WNM-3D> |
| Self-Improvement | 2026-06，Under Review | GN0 | <https://github.com/TeleHuman/GN0> |

- 2025-09 至 2026-04 的工作早于公司 2026-10 官宣成立，应视为创始团队在 TeleAI 期间的成果；本库公司路线只收署名含 Gamma Robotics 且仓库在 Gamma-Robot 组织下的 5 项（见下）。
- 官网 NEXUS 卡片标题（Cross-Domain Whole-Body Teleoperation）与 Gamma-Robot/NEXUS 仓库对应的 arXiv:2609.39000 标题（Terrain-Adaptive Teleoperation）不同，但共用项目页 nexus-humanoid.github.io。

## GitHub 组织 Gamma-Robot（2026-11-06）

- 简介：「Gamma Robotics builds self-evolving intelligence for loco-manipulation in the physical world.」；地点 China；网站 www.gammarobotics.com.cn；邮箱 contact@gammarobotics.com.cn；无公开成员；置顶 bfm_ttc、GR-BC、KODA、NEXUS。
- 公开仓库 5 个（「Showing 5 of 5 repositories」），均为 Python：

| 仓库 | 对应论文 | 许可（README） | ★ | 列表更新日 |
|------|----------|----------------|---|------------|
| [bfm_ttc](https://github.com/Gamma-Robot/bfm_ttc) | TTCO，arXiv:2610.14896 | 主代码 Apache-2.0；SPIDER sampler CC BY-NC 4.0 | 153 | 2026-10-20 |
| [KODA](https://github.com/Gamma-Robot/KODA) | arXiv:2610.21152 | 各组件保留原许可，无统一许可 | 50 | 2026-10-20 |
| [NEXUS](https://github.com/Gamma-Robot/NEXUS) | arXiv:2609.39000 | 原创部分 CC BY-NC 4.0；RSL-RL 派生模块 BSD-3-Clause | 40 | 2026-10-29 |
| [GR-BC](https://github.com/Gamma-Robot/GR-BC) | arXiv:2610.20504（CoRL 2026） | Apache-2.0 | 39 | 2026-10-17 |
| [LocoManiLab](https://github.com/Gamma-Robot/LocoManiLab) | arXiv:2610.30876 | MIT（依赖、权重、数据另计） | 12 | 2026-11-05 |

- **NEXUS：** G1 感知式全身遥操作；含地形动作适配、PPO 教师、学生蒸馏、MuJoCo / Viser 回放（`play.py` 回放 `.pt`，可导出 ONNX）与预训练 G1 ONNX 权重；训练示例 4 GPU × 16,384 环境；部署代码与论文配对语料未发布（见 [论文归档](../papers/nexus_terrain_adaptive_teleop_arxiv_2609_39000.md)）。
- **GR-BC：** README 更新记录「2026-10-16: Code released, with graph weighting, VLA training, and visualization tools.」；基于 LeRobot，支持 SmolVLA / X-VLA / π₀.₅。
- **KODA：** 公开版只含机器人运动学与运动对齐的机载视频生成；控制器动作标签、SONIC token 编码、数据导出与策略训练（M3–M5）仍为 Planned。
- **LocoManiLab：** collect → prepare → convert → train → evaluate → deploy 工作流，234 任务基准；README 写「Model integrations are retained for research and have not all been revalidated end to end」；链接 HF 数据集 `GammaRobotics/LocoManiLab`（公开、未设门槛，含 234 任务数据与部分推理 / 控制器权重，见[论文归档](../papers/locomanilab_arxiv_2610_30876.md)）。仓库 3 次提交。
- **bfm_ttc：** 冻结人形 BFM 上的测试时指令优化（TTCO），MuJoCo Warp 并行采样，G1。

## 论文署名核查（arXiv，2026-11-06）

| 论文 | v1 日期 | 第一个列出的单位 | Gamma 署名原文 |
|------|---------|------------------|----------------|
| [NEXUS 2609.39000](https://arxiv.org/abs/2609.39000) | 2026-09-30 | TeleAI | 单位 5「γ-Robotics」（Chenjia Bai 标 1,5） |
| [TTCO 2610.14896](https://arxiv.org/abs/2610.14896) | 2026-10-10（v2 10-14） | TeleAI | 单位 4「Gamma Robotics」 |
| [GR-BC 2610.20504](https://arxiv.org/abs/2610.20504) | 2026-10-14 | 浙江大学 CAD&CG | 末项「Gamma Robotics (γ)」 |
| [KODA 2610.21152](https://arxiv.org/abs/2610.21152) | 2026-10-15 | Gamma Robotics（共同一作 Zibo Zhou 的首个单位） | Zibo Zhou、Jiyuan Shi、Chenjia Bai 均标 Gamma Robotics |
| [LocoManiLab 2610.30876](https://arxiv.org/abs/2610.30876) | 2026-10-22（v4 10-28） | TeleAI | 单位 8「Gamma Robotics」 |

- 5 篇均与 TeleAI 联合署名，作者均含 Chenjia Bai（KODA 标为通讯作者）。NEXUS 9-30 已署 γ-Robotics，早于 10-20 官宣，与「近日注册成立」口径不冲突。

## 未确认 / 不收入

- 中文工商全称、具体注册日、融资金额、团队规模：未找到一手来源。
- GBot-W / GData 的发布日期与价格；PRTS「167B」数据与 95.9% 成功率仅为官网自报，无论文或评测细节对应。
- TeleHuman 组织与公司的关系官网未说明；TextOp、HUSKY 等早于公司成立的 TeleAI 工作不计为公司成果。
