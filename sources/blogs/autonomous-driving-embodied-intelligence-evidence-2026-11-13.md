# 智能驾驶四个时代到具身智能：用户分析与一手证据核对（2026-11-13）

- **类型：** personal / synthesis / 官方资料交叉核对
- **原始资料：** 用户在 2026-11-13 对话中提供的《智能驾驶的四个时代（2013–2026）→具身智能六维映射》全文（三部分、附录）
- **入库日期：** 2026-11-13
- **核对范围：** 已完整阅读用户提供的三部分、时间线、六维映射、反例、预测与附录；逐项查对下列公开的一手官方资料及部分权威新闻引用；未获得所提华泰/长江/国泰君安研报全文，未对全部二手数字作独立证实。用户原始观点与独立证实结论分开。
- **通读：** 已完成
- **一句话说明：** 把“智能驾驶历史会否在机器人重演”转为可追踪的事实 / 推论 / 待核实三层证据，不将产业推演误写成已发生事实。

## 一手来源与核对结果

| 时间 / 主题 | 可确认信息 | 一手来源与边界 |
|---|---|---|
| 2020-10 Waymo | 2020-10-08 向 Phoenix Waymo One 乘客开放完全无人驾驶服务 | [Waymo 官方](https://waymo.com/blog/2020/10/waymo-is-opening-its-fully-driverless-service-in-phoenix/)；限定服务区域，不代表全美无条件 L4 |
| 2024 FSD v12 | Tesla 公开将城市街道驾驶栈描述为端到端神经网络；“替代超过30万行 C++”为 Tesla 当时版本说明口径 | [Tesla 版本说明转引](https://electrek.co/2024/01/22/tesla-releases-fsd-v12-last-hope-self-driving/)、[Tesla 官方 FSD 定义](https://www.tesla.com/fsd-evidence-dashboard)；不等于系统所有安全/控制代码消失 |
| 2024-12 Cruise | GM 宣布不再资助 Cruise Robotaxi 开发，转向个人车辆辅助驾驶 | [GM 投资者公告](https://investor.gm.com/news-releases/news-release-details/gm-refocus-autonomous-driving-development-personal-vehicles) |
| 2025–26 中国 L3 | 2025-12-15 工信部对长安、极狐两款车型附条件许可，限定道路与速度 | [工信部原文](https://wap.miit.gov.cn/xwfb/gxdt/sjdt/art/2025/art_f734c7074580470593d4018742997640.html)；是限定场景试点，并非全部 L3 车型自由上路 |
| 2026-01/02 Mobileye | 2026-01-06 宣布拟收购 Mentee，2026-02-03 完成，约 9 亿美元交易对价 | [Mobileye 公告](https://ir.mobileye.com/news-releases/news-release-details/mobileye-acquire-mentee-robotics-accelerate-physical-ai)、[2026 季报](https://ir.mobileye.com/node/9466/ixbrl-viewer)；并不能单凭收购断言 Mobileye 黑盒业务失败 |
| 2026-03 小鹏 VLA | 第二代 VLA 2026-03-02 宣布当月推送，小鹏后续官方报道确认 3 月向用户推送 | [发布会](https://www.xiaopeng.com/news/company_news/5539.html)、[运营进展](https://www.xiaopeng.com/news/company_news/5566.html)；“L4”属于发布者技术目标措辞，非监管认定 |
| 2026-02 Waymo | 公司称每周超过 40 万次付费/运营行程，2025 年 1500 万次行程 | [Waymo 官方融资公告](https://waymo.com/blog/2026/02/waymo-raises-usd16-billion-investment-round/)；2026-03 每周 50 万单见同期媒体对 Waymo 社媒的引述，不混淆两个日期 |
| 2026 FSD 订阅 | Tesla 官网页面显示 $99/月，当前无买断销售，且 FSD (Supervised) 是 SAE L2 | [Tesla 服务页](https://www.tesla.com/support/fsd)、[Tesla 安全说明](https://www.tesla.com/fsd-evidence-dashboard)；“2026-02 具体停售日”未逐条核对 |
| 智驾/机器人算力 | DRIVE AGX Thor 开发平台标称最高 1000 INT8 TOPS；Jetson Thor 标称 2070 FP4 TFLOPS | [DRIVE AGX](https://developer.nvidia.com/drive/agx/)、[Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)；不同精度与产品指标不可直接换算 |
| 机器人 VLA | Physical Intelligence 2024-10-31 发布 π₀ 通用机器人策略 | [π₀ 官方技术博客](https://www.pi.website/blog/pi0)；不能等同于车端 VLA 的动作/安全接口 |
| 物理世界共用底座 | NVIDIA 2026 年 Cosmos 3 官方将机器人、自动驾驶和视觉代理列为目标场景 | [NVIDIA 官方新闻](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai)；平台共享不等于模型在两个场景直接零样本迁移 |
| Robotaxi 系统性风险 | 2026-03-31 武汉多辆“萝卜快跑”停在道路上，武汉交警称初判系统故障、无人受伤 | [武汉交警原通报转载](https://www.peopleapp.com/column/30051781800-500007421664)；警方未给出具体车辆数量或“暂停发放 L4 牌照”结论 |

## 核对中需要纠偏的叙事

1. **汽车路线不是四段整齐接棒**：模块化、BEV、端到端、世界模型/VLA 在 2026 年并存；高精地图是否必需、端到端范围、监管等级与模型范式是不同轴。
2. **渐进 L2 并未“全灭” L4 原生路线**：Waymo 已实现限定运营区域的付费规模化；Argo / Cruise 的退出也不能推导为其他技术路线必然失败。
3. **Mobileye 不是已证实的“黑盒失败”案例**：Mentee 收购说明业务拓展，不能当作其汽车收入或商业模式失效的直接证据。
4. **数据成本/频率不可简单量纲比较**：监督驾驶里程、具身操作小时、动作频率与有效学习样本无统一换算。
5. **利润、出货、估值、未来门槛应独立验证**：45 万元两班 3.3 年回本、2028 通用双足回本、100 亿英里、Waymo 3000+、纯大脑无法独立销售、2026 年 Cybercab 9 月正式载客等未获得完整相应一手证据，wiki 仅把它们作为待核实假设，不直接引用为结论。
6. **GPT-6 Astra 120 项仿真抓放任务胜过专用模型**：可能对应仓库既有 [评测项目](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md)，本次未重读该评测原文及实验设置，不能仅凭研报转述泛化到工业机器人能力。
7. **上海人形机器人 L1–L4、智元 G1–G5**：不同发布主体的分级方案不能与 SAE 自动驾驶法律责任分级划等号；本次未取得它们的完整标准文本。

## 对 wiki 的映射

- 主对照知识节点：[智能驾驶→具身智能：2013–2026 六维对照](../../wiki/comparisons/autonomous-driving-embodied-intelligence-2013-2026.md)
- 现有知识：[端到端驾驶十大路线](../../wiki/overview/e2e-autonomous-driving-top10-algorithms.md)；[机器人公司基础模型路线对照](../../wiki/comparisons/robot-foundation-model-company-paths-2026.md)
