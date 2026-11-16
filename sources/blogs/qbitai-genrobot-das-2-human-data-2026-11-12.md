# 大模型都能控机器人，Human Data 还得怎么卷？

> 二手报道归档：量子位（QbitAI），田晏林，2026-11-12。本文仅提炼技术线索，不转载原文全文。

- **原文：** <https://mp.weixin.qq.com/s/S_Zx0OdoY3V9bGSy8eQ1Ew>
- **用户提供：** 同名 PDF，16 页，2026-11-14 保存的公众号页面
- **类型：** 行业报道 / 产品介绍（非同行评审论文）
- **核查日期：** 2026-11-14
- **核对范围：** 通读 PDF 第 1–16 页，覆盖 Human Data 缺失模态、Ego/Wrist/Dex 配置、传感规格、时空对齐、视觉蒸馏设想及 Whole-body。
- **通读：** 已完成
- **官方核验：** [DAS Ego 2.0](https://www.genrobot.ai/products/ego2)、[DAS Dex 2.0](https://www.genrobot.ai/products/dex2)、[DAS WristCam](https://www.genrobot.ai/products/wristcam)

## 核心线索

1. **Human Data 不只是视频小时数。** 第一视角覆盖任务语境，但手部遮挡、接触时刻、接触力以及走近/调整身体/搬运的前后链路经常缺失。
2. **按任务分层采集。** Ego 提供环境和过程；Ego + Wrist Camera 更接近操作区域；Ego + Dex 加入手部关节与原生触觉。报道将 Ego + Wrist Camera + Dex 三模态配对采集描述为**正在探索**的路线，不能当成已验证的视觉触觉模型。
3. **Ego 2.0。** 报道列出 6 颗全局快门摄像头、30 fps、340° 水平覆盖、2×5 MP + 4×2 MP、4 mm 标称手部跟踪；磁吸电池热换、2.5–3 小时续航和 MCAP；核心性能参数可由官方产品页交叉核对。
4. **Dex 2.0。** 侧置外骨骼减少指腹/掌面干扰，单手 21 个关节编码器、980 个触觉单元、0.1 N 最小检测力、1 mm 触觉空间分辨率、100 Hz 触觉采样，均有官方规格支持。
5. **质量关键是同步。** 头手约 1 ms 标称对齐；按手部速度 0.5 m/s，1 ms 时差等于约 0.5 mm 的运动位移，仅是几何估算，**不是设备整体定位误差**。
6. **Whole-body 不是手部片段的替代品。** 报道按策略（靠近/站位）、协调（躯干–肩臂–手）、执行（起身/转向/搬运）讨论完整人类示范与跨机器人本体映射。

## 待验证与口径边界

- 报道称单季度「有效数据」交付从十万小时级增至百万小时级；本次官方产品规格页未披露可独立核验的交付明细、质检阈值或客户验收记录，**仅作公司/媒体口径**。
- 报道提到连续一个月相同动作触觉响应一致，以及 Simate、自然意志、Dreamlabs、生数 Vidu AI 等模型中的验证；当前未见同一口径的公开独立测试协议，不能解读为可复现的基准。
- 视觉推断接触/触觉的配对监督、蒸馏和未来推断均属设想；隐藏接触、重量、力本身不可保证由 RGB 唯一恢复。
- 厂商现有开放的 **Gen-HumanEgo / das-datakit** 不等于已经发布 **DAS 2.0 全模态数据集或端到端训练代码**；设备代际、授权和实际输出字段须分开核对。

## 对 wiki 的映射

- [DAS 2.0（Ego / Dex / WristCam）技术体系](../../wiki/entities/genrobot-das-2-0.md)
- [Gen-HumanEgo（上一代 DAS-Ego）](../../wiki/entities/gen-human-ego-dataset.md)
- [Ego 数据采集技术索引](../../wiki/overview/ego-category-01-data-collection.md)
