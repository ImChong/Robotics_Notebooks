# Kyber Labs（kyberlabs.ai）

- **类型**：公司官网 / 产业侧原始资料汇编
- **URL**：<https://kyberlabs.ai/>
- **收录日期**：2026-06-25
- **说明**：以下条目为 **2026-06-25** 可访问的官网页面摘录；硬件规格与路线图以官方后续披露为准。官网为 GoDaddy Website Builder 静态站，正文以 Home / Demos / FAQ 等子页为主。

## 一句话

Brooklyn 初创 **Kyber Labs** 面向 **具身 AI（embodied AI）** 构建 **双臂 + 仿人灵巧手** 操作平台：强调 **全背驱动（backdrivable）、力矩透明（torque-transparent）、机械柔顺** 与 **低成本**，服务高混低量装配与实验室自动化，**不做整机人形**。

## 为什么值得保留

- **灵巧手产业对照样本**：与 [Wuji Hand](../../wiki/entities/wuji-robotics.md)、[Allegro Hand](../../wiki/entities/allegro-hand.md) 等并列，代表「**为 AI 控制与大规模数据采集而设计**」的硬件路线。
- **背驱动 + 电流本体感知叙事**：公开 demo 将 **接触检测、柔顺交互、螺纹/fastener 装配** 与 **驱动电流 proprioception** 绑定，可交叉 [Humanoid 执行器 102 · 柔顺与感知](../../wiki/overview/humanoid-actuator-102-compliance-sensing.md)。
- **反人形定位清晰**：FAQ 明确「价值在人形机器人的手，而非双腿」，适合与全身人形整机厂对照阅读。

## 官网核心信息（2026-06-25 摘录）

### 首页（`/`）

- **定位**：「Robots designed for AI」— 自底向上为 **AI based controls** 设计的机器人操作平台；硬件模仿人类 **机械柔顺与精细运动**，支撑规模化精细操作。
- **愿景**：当前机器人系统难以支撑 AI 带来的多样化用例；需在 **硬件与软件** 上范式转移。
- **产品方向**：突破性 **双臂（bimanual）操作平台**，面向 **embodied AI**，在 **非结构化环境** 中以最小部署成本完成装配与操作。
- **方法**：通用灵巧手是机器人能力瓶颈；自研新手以支撑 **装配与制造任务的大规模 AI 训练数据采集**。
- **团队**：
  - **Tyler Habowski**（Cofounder）— SpaceX 老兵（F9/FH/Starship 可复用机构与制造方法）；曾在 Machina Labs 任 ML Engineer。
  - **Yonatan Robbins**（Cofounder）— 工业设计师（微外科器械至影视）；曾在 Tarform 领导机械团队打造电动摩托车。
  - **Julian Viereck**（Robotics Research Scientist）— 前 Amazon ML Engineer；NYU PhD（足式运动与灵巧控制）；曾参与 Google X、CERN 项目。
- **投资人（首页列举）**：Cortical Ventures、Starburst Ventures、Elliptic Ventures、Earthrise Ventures、Fundomo、Heuristic Capital、Trevor Blackwell。
- **联系**：hello@kyberlabs.ai；地址 19 Morris Ave, Building 128, Newlab - Kyber Labs, Studio 204, Brooklyn, NY 11205；Twitter **@KyberLabsRobots**。

### Demos（`/demos`）

| Demo | 要点 |
|------|------|
| **临床病理实验室协作** | 与病理实验室合作；单系统完成真实实验任务（工具使用、精细操作、高层规划）；**一镜到底、无遥操作**；**skills based AI** 兼顾泛化与确定性可靠工作流 |
| **固体称量转移** | 以食盐模拟粉末/颗粒/凝胶等 **固体转移**；面向自动化湿实验流程 |
| **100 次随机 vial 循环** | 取管、开盖、旋紧密封、归位托盘；强调 **螺纹盖** 为接触丰富、高精度难题 |
| **自研触觉传感器原型** | 检测敲击、运动、多点接触与力幅；**廉价、可更换**；demo 尚未使用，与 **力矩透明关节** 组合以低成本恢复丰富接触 |
| **SpaceX 风格装配序列** | 连续完成 pick-place、插入、螺母旋紧、手内操作；面向 **低产量零件** 无需专用产线、可快速改线 |
| **高速螺母旋紧** | 大螺母在螺栓上 **实时高速旋转**；依赖 **背驱动 + 力矩透明** 顺应几何；重点在可靠性与能力而非裸速度 |
| **羽毛接触停指** | 仅靠 **直驱作动物理** 在接触羽毛时停指；**背驱动 + 力矩透明** 通过驱动电流感知接触 |
| **浆果快速抓取** | 四颗软果快速拾放不压碎；柔顺与力矩透明简化接触控制 |

### FAQ（`/faq`）

- **建什么**：机器人操作平台，从 **高混低量**、传统自动化难覆盖的工作起步；核心是 **双臂 + 仿人手**，为 embodied AI 设计。
- **为何聚焦手**：灵巧、鲁棒的手是通用机器人瓶颈；没有手，AI 难做真实世界工作。
- **手的差异**：**背驱动、柔顺、可负担、专为 AI 控制设计**。
- **是否做人形**：**不完全是** — 人形价值在 **手能做什么**；可固定工位、导轨、轮式或腿式底座，灵巧操作是通用机器人关键。
- **成本**：不计划直接卖手，但硬件从设计起优先成本，目标是 **数百美元级而非数千美元**。
- **早期任务**：装配、机床上下料及其他高混低量工业任务。
- **目标**：不是「史上最好手」，而是 **最有用的手**；夹爪适合 pick-place，手解锁复杂操作与高价值装配制造。
- **硬件还是软件**：**两者** — 硬件赋能软件，终局是 **AI 驱动操作**。
- **触觉**：开发 **低成本、高密度、可扩展** 指尖传感器。
- **与人形初创差异**：不直奔「放进家里的通用机器人」；**务实分步**，以 **成本 + embodied AI** 为核心，先聚焦手。

## 二手公开报道（策展索引，非官网正文）

以下用于补全团队背景与硬件细节线索，**工程引用仍应回链官网 demo/FAQ**：

- Humanoids Daily · Kyber Labs 背驱动灵巧手报道：<https://www.humanoidsdaily.com/news/kyber-labs-emerges-with-a-high-speed-backdrivable-robotic-hand>
- Humanoids Daily · Over the Horizon 播客深访：<https://www.humanoidsdaily.com/news/kyber-labs-founders-crash-the-over-the-horizon-podcast-for-a-technical-deep-dive>（提及 **20 DoF / 40 腱 / 前臂集成的无框 BLDC**、计划向研究者免费提供约 50 只手等，需以官方后续为准）
- LinkedIn 公司页：<https://www.linkedin.com/company/kyber-labs>

## 对 wiki 的映射

- 升格页面：[wiki/entities/kyber-labs.md](../../wiki/entities/kyber-labs.md)
- 交叉更新：[wiki/overview/notable-commercial-robot-platforms.md](../../wiki/overview/notable-commercial-robot-platforms.md)、[wiki/entities/wuji-robotics.md](../../wiki/entities/wuji-robotics.md)、[wiki/entities/allegro-hand.md](../../wiki/entities/allegro-hand.md)

## 参考链接（索引）

- 官网首页：<https://kyberlabs.ai/>
- Demos：<https://kyberlabs.ai/demos>
- FAQ：<https://kyberlabs.ai/faq>
- Want a hand?（合作意向）：<https://kyberlabs.ai/want-a-hand%3F>
- Contact：<https://kyberlabs.ai/contact-us>
