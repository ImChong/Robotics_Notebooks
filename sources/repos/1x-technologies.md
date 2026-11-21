# 1X Technologies（原 Halodi Robotics）

> 来源归档

- **标题：** 1X Technologies
- **类型：** repo / company
- **官方站点：** https://www.1x.tech/
- **关于页：** https://1x.tech/about
- **NEO 产品页：** https://www.1x.tech/neo
- **入库日期：** 2026-05-07
- **核对范围：** 通读 1X 官方站点 `https://1x.tech/about`、`https://www.1x.tech/neo`（技术规格与发布时间线）以及 `1x-technologies/1xgpt` 公开仓库 README，结合第三方背景资料核对 Halodi→1X 沿革（2014 成立、2018 Revo1 电机、2022 EVE 工厂部署、2024 NEO Beta、2025 NEO Gamma、2026 Early Access 计划）及 NEO 公开硬件指标；Redwood / NEO 生产固件与商业交付条款未公开，未复现或验证实机性能。
- **通读：** 已完成
- **一句话说明：** 挪威–美国背景的通用人形机器人公司；产品线包含轮式人形 **EVE** 与面向家庭场景的双足 **NEO**，强调真实部署数据、肌腱驱动（Tendon Drive）安全硬件与 AI（Redwood / 1XWM）结合。
- **沉淀到 wiki：** 是 → [`wiki/entities/1x-technologies.md`](../../wiki/entities/1x-technologies.md)

## 官方公开时间线与 NEO 规格摘录（核查自 `1x.tech/about` 与 `1x.tech/neo`）

- **公司演进：** 2014 年作为 Halodi Robotics 创立；2018 年推出 Revo1 低减速比肌腱驱动伺服；2022 年轮式人形 EVE 部署至工厂；2023 年转向家用双足研发；2024 年推出 NEO Beta；2025 年更新 NEO Gamma 并开启早期家用试用；2026 年计划推进 Early Access 交付。
- **NEO 公开硬件指标（官网规格表）：** 身高 `5'6"`、自重 66 lbs（约 30 kg）、举升 154 lbs / 搬运 55 lbs / 单臂负载 18 lbs；自由度：双手 22×2、双臂 7×2、颈部 3、脊柱 2、双腿 6×2；步行 1.4 m/s、最大奔跑 6.2 m/s、手部最高速度 8.0 m/s；842 Wh 电池续航 4 小时；肌腱驱动回驱率 95%、声噪 22 dB、3D 晶格软体包覆（HIC < 250，无夹手点，手部 IP68）；算力平台为 1X NEO Cortex（NVIDIA Jetson Thor，官网标称最高 2070 FP4 TFLOPS），配备双目 8.85 MP 90 Hz 鱼眼相机。
- **交互与人机协同：** 支持机载 Redwood AI 自主执行家务，亦提供预约「1X Expert」远程示教/遥操作指导未见任务，并配 25-DoF 肌腱驱动手更新（2026-07-09 新闻）。

## 第三方综述（便于核对时间与融资脉络）

- Wikipedia — [1X Technologies](https://en.wikipedia.org/wiki/1X_Technologies)（公司沿革：2014 年以 Halodi Robotics 创立等叙述）
- IEEE Robots Guide — [EVE](https://robots.ieee.org/robots/eve)（产品条目）

## 备注（维护者）

- 商业条款、交付时间与融资细节变化快；规格与路线图以官网与官方新闻为准。
- 公开报道曾提及 OpenAI Startup Fund 参与早期融资；具体轮次与金额请交叉核对 SEC 披露或权威新闻稿。
