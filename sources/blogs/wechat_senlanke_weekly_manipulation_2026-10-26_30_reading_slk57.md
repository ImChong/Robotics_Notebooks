# senlanke 周报 10-26–10-30 Manipulation 57 篇 通读

> 来源归档（原文通读 · 支撑 `wiki/overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md`）

- **标题：** 【10.26-10.30前沿论文动态】Manipulation 57篇
- **作者：** senlanke / 具身运控lab（微信公众号）
- **类型：** blog / weekly-digest
- **原文：** <https://mp.weixin.qq.com/s/uIziIlqQrYGvUp-BivNHPw>（2026-11 用 `scripts/fetch_wechat_article.py` 取得全文，约 2.1 万字符）
- **一句话说明：** 按 10 个分栏列出 57 篇 Manipulation 论文，每条含英文标题、中文标题、机构、论文链接、「解决的问题」「创新点」。
- **核对范围：** 通读公众号全文 10 个分栏、57 个条目的「解决的问题 / 创新点」及文末；未逐篇打开各 arXiv 原文（技术事实以各详情节点引用的 arXiv 原文为准），机构信息为周报所述、未核实。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **分栏与篇数：** VLA / 通用操作策略 7；WAM / World Model 10；在线/离线 RL 与后训练 4；模仿学习与生成式动作策略 6；双臂 4；灵巧 4；柔性/可变形 3；人形上半身 3；长程任务、推理与规划 7；数据集、预训练、跨本体与硬件 9；合计 57。
- **VLA 栏主线：** 未来预测作为训练期辅助监督、推理期可移除（PDRVLA 预测未来 RGB / 深度 / 光流特征；SaiVLA-JEPA 训练期加未来特征 Head，成功率约 71.6% 到 82.5%）；冻结 VLA 上的外挂模块（R2A 的运动基元图检索、VLA-Stem 的 RL 安全策略）；端侧部署的混合精度量化（QbD-VLA，权重平均约 4 bit）。
- **WAM 栏主线：** 跨本体统一动作接口（RynnWorld-Latent、TAPESTRY、CoAWM 的潜在 / 连续动作）；以 3D 运动代替完整视频预测（Trace-WAM、WorldFlow）；压缩与加速（GAWM 参数量低于 100M；OSWAM 一步扩散，较直接策略式 WAM 约快 3 倍）；统一视频与 4D 感知（GenSight）。
- **RL 栏：** Echo 的单阶段 off-policy 视觉 RL；Event2Reward 的进度前进 / 停滞 / 倒退奖励；FERRET 用环境编辑消除奖励特征歧义；物体中心 RL 的卡扣拆解。
- **生成式策略栏：** 任务专用 Flow 源分布与调度（TOPS）、Flow 策略高斯读出（GAR，约 0.02% 训练状态拟合输出层）、二值可行性安全修正、示范扩展（DEED）、数据驱动预测控制（RoKON）。
- **双臂 / 灵巧 / 可变形 / 人形栏：** GPU 冲突搜索多臂规划、摩擦裕度的非抓取双臂操作、跨灵巧手本体预训练、双臂失败诊断；手内位姿 RL + 残差安全过滤、运动学-触觉重定向、动态接物；因子图受力估计、形变语义失败、形状伺服教程；Real-to-Sim-to-Real 数据、可提示全身执行层、全身操作。
- **长程 / 评测栏：** 任务图验证与局部修复（LOGIV）、进度监测（ProLoop、ReSTEP / REMORA）、免训练自我改进（RoboDoctor）、Coding Agent 从失败构建 RL 任务（ELEVATE）。
- **数据 / 硬件栏：** RoboAtom 原子动作监督；EgoContact 22,500 个事件，抓取成功率 16% 到 58%；DexSuite（30 任务、10 臂、12 末端）；UMI-Live（40 小时数据）；VLA-HWBench 在 Jetson AGX Thor 与 AMD Ryzen AI Max+ PRO 395 评 18 个开源 VLA，动作生成方式比参数量更能解释延迟差异；EBench 与 CuraEval 评测；DEX-PHD、Tactus（47 触觉位置、16 个 IMU、指尖误差约 11.0 mm）采集设备。
- **局限（原文自身）：** 周报仅给「问题 / 创新点」摘要，部分条目（如 E0.5、TAPESTRY、TouchBridge、FLoRe）描述较概括；DEX-PHD 机构标注「待核实」。

## 对 wiki 的映射

- [`wiki/overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md`](../../wiki/overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md)
- 逐条目条目索引见 [wechat_senlanke_weekly_manipulation_2026-10-26_30.md](wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
