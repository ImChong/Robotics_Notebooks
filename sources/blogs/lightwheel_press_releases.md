# Lightwheel 官网 Press Releases 归档（2026-10-10 共 5 篇）

> 来源归档（press release / Lightwheel 官方）

- **列表页：** <https://lightwheel.ai/blogs>（点击「Press Releases」筛选，客户端渲染；筛选后地址为 `/blogs/releases`）
- **入库日期：** 2026-10-10
- **抓取方式：** Playwright（Chromium headless）打开列表页、点击「Press Releases」读取卡片标题与日期，再逐条点击进入详情页读取 `innerText`；列表未见分页
- **覆盖 wiki：** [Lightwheel 公司页](../../wiki/entities/lightwheel.md)「新闻稿要点」小节

## 列表

| 列表日期 | 标题 | 链接 |
|----------|------|------|
| 2026-07-09 | Lightwheel and PICO Partner to Advance Human Data Collection Infrastructure for Physical AI | <https://lightwheel.ai/media/lightwheel-pico-partnership> |
| 2026-07-03 | Lightwheel and MANUS Announce Strategic Partnership to Build the Data Infrastructure Backbone for Physical AI | <https://lightwheel.ai/media/lightwheel-manus-partnership> |
| 2026-05-06 | $100M in Q1 Orders — Lightwheel Marks the Start of Physical AI at Scale | <https://lightwheel.ai/media/q1-orders-physical-ai> |
| 2026-04-14 | Martin Elbs Joins Lightwheel as VP of Global Sales | <https://lightwheel.ai/media/martin-elbs-vp-global-sales> |
| 2026-04-09 | Lightwheel and PeritasAI Announce Strategic Partnership to Bring Physical AI into Perioperative Workflows | <https://lightwheel.ai/media/lightwheel-and-peritasai-announce-strategic-partnership> |

PICO 与 MANUS 两篇正文电头只写「July 2026」，具体日以列表卡片为准。

## 逐篇要点（归纳）

### $100M in Q1 Orders（2026-05-06）

- **自报：** 2026 年 Q1 签下约 **1 亿美元订单**，覆盖仿真、数据生成、评测与面向部署的系统；称「不是财务里程碑，而是行业转向的证据」。未披露客户、确认收入口径或审计信息。
- **客户两类：** 前沿 Physical AI 模型团队（瓶颈在数据质量 / 多样性 / 物理真实性，需要持续数据基础设施）与工业企业（瓶颈在任务能否被训练、验证和部署后改进）。
- **四阶段叙事：** World（扫描重建工位、料盘、传送带并做物理准确仿真）→ Behavior（EgoSuite 第一视角人类示范 + 大规模数据生成 + 持续训练）→ Evaluation（RoboFinals 大规模仿真诊断）→ Deployment（先上最高频、最可预测的子任务，再扩展任务包络，真实数据回流形成飞轮）。
- **生态（自报）：** 受邀以 core advisor 身份加入开源 GPU 物理引擎 **Newton**（与 NVIDIA、Google DeepMind、Disney Research、Toyota Research Institute 并列）；Lightwheel 开发的 **LeIsaac** 被 Hugging Face 官方文档采纳为具身仿真标准框架。
- 自称「唯一为闭环 Physical AI 而建的公司」——营销表述。

### Lightwheel × PeritasAI（2026-04-09）

- PeritasAI 为围术期（perioperative）智能编排公司，平台名 **PeriVerse**；Lightwheel 提供仿真、Real-to-Sim / Sim-to-Real 管线、合成与第一视角数据、训练与评测。
- **公告数字：** 多阶段计划估算 **5,600 万美元**，目标在 **2026–2027 年** 让 **最多 200 台人形机器人** 开始部署进围术期工作流；初期试点已与部分医疗系统和 OEM 伙伴开展。
- Lightwheel 发言人：Louis Lian（VP of Partnerships and Strategy）。

### Martin Elbs 加入任 VP of Global Sales（2026-04-14）

- 20 年以上汽车 / 工程 / 工业市场经验；此前任 **IPG Automotive** SVP 兼首席商务官，曾在 **Lotus Engineering**、**ETAS** 任职。
- 负责全球销售，重点 OEM、汽车与工业企业，**尤其欧洲**等战略市场；Steve Xie 在稿中署名「Founder and CEO」。

### Lightwheel × MANUS（2026-07-03）

- MANUS（荷兰埃因霍温，2014 年成立）提供高精度数据手套：每只手 **25 自由度**、毫米级精度；稿称其为 NVIDIA Isaac Teleop 官方数据手套。
- 协议由 Lightwheel **联合创始人兼总裁 Haibo Yang（杨海波）** 与 MANUS 联合创始人兼 CEO Stephan van den Brink 签署；内容含技术集成、联合市场推广与联合开发。
- Lightwheel 正在建设 **Human Data Capture Platform（HDCP）**——标准化人类数据采集的开放平台，MANUS 为核心采集伙伴；Lightwheel 称可通过合成数据生成把每条示范 **放大 100–1,000 倍**（自报）。
- 稿称 Lightwheel 为 **Newton Technical Steering Committee** 成员，并「defines the SimReady standard」（自报）。
- **About Lightwheel：** 「Founded in 2023 and headquartered in Santa Clara, California」；产品线 SimReady / EgoSuite / RoboFinals；SimReady 资产收录于 NVIDIA Isaac Sim。Steve Xie 署名「Co-Founder & CEO」。

### Lightwheel × PICO（2026-07-09）

- 与 XR 厂商 **PICO** 组建 **联合产品团队**，共同开发下一代通用人类示范数据采集硬件；PICO 出硬件研发、量产与供应链，Lightwheel 出人类数据系统、场景定义与行业方案。
- 稿中称 Lightwheel 的自研全栈仿真平台名为 **SimFoundry**（物理求解、真实测量、资产生成）。注意与 NVIDIA NVlabs 的同名论文 / 仓库 SimFoundry（arXiv 2606.28276）**不是同一事物**（据名称与描述判断，推测仅为同名）。
- About Lightwheel 段称其为「closed-loop Real2Sim2Real platform」，并自称「global leader in human data, simulation evaluation, and real-world deployment」。

## 可信度与使用边界

- 全部为公司新闻稿：订单额、项目金额、机器人台数、放大倍数均为 **自报** 或合同目标，非交付结果。
- 「Q1 订单 1 亿美元」与中文媒体报道的营收说法（36 氪转载铅笔道：「过去一年营收 10 倍增长，Q1 营收有望超过去年全年」）口径不同，不要互相换算。
