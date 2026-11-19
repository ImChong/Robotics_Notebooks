# 具身机器人 FDE / Deployment：厂商一手资料核对（2026-11-19）

- **类型：** sites / 官方技术发布与产品文档对照
- **核对范围：** 逐项阅读下列厂商官方文章、产品文档与声明中关于部署流程、开放方式、交付阶段、收费/风险的相关章节（不包括未公开的合同及现场验收数据）；Agility SEC 只核对 CAP / 运行时数相关披露段落，不宣称通读整份 SEC 申报书。
- **通读：** 已完成
- **核对日期：** 2026-11-19
- **用途：** 一手证据归档；文章源另见 [微信行业综述](../blogs/wechat_embodied_ai_heart_robot_fde_delivery_20261119.md)。

## 一手资料及已核验边界

### 1. 智元 AGIBOT：Genie Studio 2.0
- **官方原文：** https://www.agibot.com/news/newsroom/post-130 （2026-10-26）
- **事实：** Flywheel（数据/模型）、Agent（技能及任务编排）、Monitor（设备状态/监控/运维）；常州 6 台 CNC + AGV，需求到产线验证 14 天，接入 PLC、MES、视觉与工装，流程/参数/接口模板化。
- **证据边界：** 厂商案例与单项目时长，不足以证明所有部署都在 14 天内完成，更不证明大规模独立集成商交付。

### 2. 破壳 PokeBot：RoboAPI
- **官方开发文档：** https://developers.pokebot.com/zh/docs/roboapi/latest
- **官方项目说明：** https://pokebot.com/zh-cn/ai
- **事实：** 提交相机画面、机器人状态、任务指令给 Poke-Aurora；可用本地 Linux 运行程序/控制台或调用云 API。
- **证据边界：** 技术预览/厂商展示，不将零样本泛化当作所有机器人上线成功的保证；非开放模型权重声明。

### 3. 原力灵机 Dexmal：MaaS
- **官方平台：** https://maas.dexmal.com/
- **官方模型/部分开源：** https://github.com/dexmal/opendm
- **事实：** DM0.5 MaaS 介绍模型调用与 SFT、DAgger、RECAP 后训练方案，DexOS Lite 用于机器人连接；官方 opendm 仓库公开部分 SFT、评测及模型资产。
- **证据边界：** MaaS 云服务与仓库公开部分能力不同；不得将线上托管服务误写为完整自托管开源平台。

### 4. Physical Intelligence：Pi API beta
- **官方原文：** https://www.pi.website/blog/api-beta （2026-10-13）
- **事实：** 上传本体数据 → 在平台管理的算力上微调 → 经托管推理网关控制机器人；Almond 案例描述 4 天迭代完成去毛刺模型；厂商说明云端推理存在网络连接挑战。
- **证据边界：** beta 伙伴案例不等于工业规模化保证；没有从本文资料核得可复用的统一 SLA/单价。

### 5. Dream Labs：Dream-1 / FaaS
- **官方发布原文：** https://dreamlabs.ai/research/dream-1/ （2026-10）
- **官方客户端：** https://github.com/dreamlabs-co/dreamlabs
- **事实：** Dream-1 研究 beta 的公共动作 API；FaaS 按开发者演示数据适配；12 家 beta 伙伴平台展示、12 台机器人中 10 台语言跟随分数在其评测中改善；两个 5 分钟样例，亦有 77 分钟、98 分钟的适配案例。
- **证据边界：** 不同机型所需数据量不同；官方指出模型未通过安全关键用途验证，且 IK/本体差异影响效果；客户端开源不等于 Dream-1 模型权重开源。

### 6. Formic：Full Service / Humanoid pilots
- **官方原文：** https://formic.co/solutions/humanoids
- **事实：** 固定月费、客户不承担前期资本支出；包括数据采集/训练/部署/持续微调/7×24 监控和按需遥操作；人形试点限定单任务/单工位/单班次，约 60–90 天到首次生产周期、额外 30–60 天扩展（官方预期）。
- **证据边界：** 官方明确表示人形仍处于窄场景试点，不能把其描述为已形成普适的人形量产部署。

### 7. Agility Robotics：Digit CAP（SEC 披露段落）
- **一手文件：** https://www.sec.gov/Archives/edgar/data/2074973/000121390026118037/ea0311221-s1_agility.htm
- **事实：** CAP 通过技术验证 → 现场 PoC → 商业 RaaS 逐步推进；早期披露参与费约 50 万美元。公开文件称截至 2026 年 5 月 Digit v4 在 9 处客户设施已部署或承诺部署、累计超过 65,000 小时。
- **证据边界：** “已部署或承诺部署”不是“全部已投产”；CAP 费用不是每台机器人价格，也不能代表普遍合同结构。此节为 SEC 目标段落复核，不称通读整份 S-1。

## 尚未独立验证的文章陈述

- 原力无限 CTO Deployment Scaling 的原始演讲、Tesla 内部制造部署的具体组织流程；
- NCS/安徽华智天成三方履约规模、擎天租城市覆盖与 WAIC 台数、公共中试平台工程服务；
- 各企业跨项目部署成功率、长期毛利及可复制成本曲线。

这些内容不作为“已验证”数值填入 wiki 选型表。

## 对 wiki 的映射

- [具身机器人 FDE 交付对比](../../wiki/comparisons/embodied-fde-deployment-business-models-2026.md)
