# WayTrace: A Map-to-Language-to-Trace Hierarchy for Embodied Urban Navigation / WayTrace：面向具身城市导航的地图—语言—轨迹分层架构

> 来源归档（paper）；本次以论文原始 HTML v1 为准，取代此前仅凭公众号二手转述的索引级摘录，不转载论文全文。

- **标题：** WayTrace: A Map-to-Language-to-Trace Hierarchy for Embodied Urban Navigation
- **中文标题：** WayTrace：面向具身城市导航的地图—语言—轨迹分层架构
- **作者：** Manthan Patel, Jonas Frey, Pascal Roth, Marco Hutter, Cesar Cadena
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.05129v1>
- **HTML 全文：** <https://arxiv.org/html/2611.05129v1>
- **PDF：** <https://arxiv.org/pdf/2611.05129v1>
- **项目主页：** <https://way-trace.github.io/>
- **版本：** v1，2026-11-03 16:13:42 UTC；论文标注 Under Review，不能当作已正式录用
- **最初入库日期：** 2026-11-05
- **原文复核日期：** 2026-11-07
- **核对范围：** arXiv HTML v1 正文 I–VII（含 Fig. 1–10、表 I–IV）与附录 A–F（实现、数据、基准及补充表 V–XI、模型提示词）；逐项对照方法、数字、局限、发布声明。
- **通读：** 已完成

## 核心贡献（论文 III / V）

1. **地图到语言：** Google Routes API 返回步行路线，经 OpenStreetMap 行人路网调整，GNSS 仅定位大致路线步骤；远端 VLM 根据导航步骤、50 m 朝向对齐卫星图与最近 15 m 相机历史，每 2 s 选择一条原子指令。
2. **语言到图像轨迹：** RynnBrain-9B 基座的车载 trace predictor 观察当前帧、5 帧历史、过去约 16 m 的行走示意图与原子指令；先预测可见终点，再预测 10 个归一化像素轨迹点。部署时以 goal token 隐状态接 MLP 一次调用回归轨迹。
3. **图像轨迹到动作：** 双目深度把像素轨迹投影到 3D，地面过滤后在里程计坐标系锚定路径，4.5 m 前视 carrot 交给 5 Hz 深度输入的 RL 导航策略；高低层异步，旧路径持续执行直到新轨迹就绪。
4. **自动监督数据：** 51,907 条 Mapillary 序列（182 城市/地区的来源池）经 VGGT-Ω 重建、未来足迹投影、VLM 指令标注与审核，形成约 288k reconstructed-walk 样本；同图像多指令的 teacher 伪标注与历史传播形成约 863k intent-conditioned 样本。合计 1,190,594 条（含重复抽样的人工/NaviTrace 样本）。
5. **模型训练：** trace predictor 采用有监督微调/LoRA；单图 teacher 曾使用 GRPO，**最终部署 trace predictor 并未采用 GRPO 训练**，切勿混淆。

## 评测协议与论文原始数值（IV–V / 附录 C–E）

| 设置 | 作者报告结果 | 口径 |
|---|---|---|
| Isaac Sim 5.0 苏黎世旧城 | 10 条路线 152–864 m，共 4.68 km；每种 GNSS 设置 30 次运行 | 模拟 Unitree B2W；场景无行人/车辆 |
| 完美 GNSS，长程成功率 | WayTrace **83.3%**；CityWalker **20.0%**；VAMOS **13.3%**；OmniVLA **0%** | Table I；含过街规则判定，非单纯终点到达率 |
| 7 m RMS 漂移 GNSS | WayTrace **63.3%**；CityWalker + 同一 RL 策略 **23.3%**；VAMOS + 同一 RL 策略 **10.0%** | V-B / Table VIII；后两者使用作者统一的 lifting/RL 后端 |
| 完美 GNSS 的人行横道违例 | WayTrace **2.2%**，CityWalker + RL **54.8%**，VAMOS + RL **31.8%** | Table VII；不能把路口违例率与全任务成功率混同 |
| Human-Test | NaviTrace 分数 **83.2**；ADE **5.0**（图像尺度百分比）；转向侧正确率 **97.9%** | Table III；953 条人工轨迹 / 332 张图，open-loop 双阶段模型 |
| Walks-Test | NaviTrace 分数 **70.4** | Table III；31,012 条自动标注样本 |
| 时序历史消融 | 有历史 **94.7%**；仅当前帧 **74.7%** | V-D / Table XI；50 个短任务 ×3 seeds，不是长程导航成功率 |
| 意图数据消融 | 327.8k 数据 NT **79.33** → 加 863k 意图数据 NT **83.32** | Table X；同设置下的训练数据规模比较 |
| 编排器关键决策错误 | Qwen3.8-Flash-Next **5.1%** | Table IV；957 条 replay 决策 |

### 真机四条路线（Table II）

| 目的地 | 路线长度 | 均速 | 人工指令改写 | 论文标记的自主里程占比 |
|---|---:|---:|---:|---:|
| Starbucks | 426 m | 1.12 m/s | 0 | 100% |
| Snipes | 248 m | 1.01 m/s | 0 | 100% |
| Dieci | 312 m | 1.11 m/s | 0 | 100% |
| Mendokoro | 1,368 m | 1.19 m/s | **2** | **97.2%** |

**关键边界：** Starbucks 路线操作员为确认电车安全暂停过机器人，但未改指令/路径，故作者未计作 intervention；不可将表中 100% 解读为绝对无人工监控。1.37 km 的 Mendokoro **不是零干预**，两次覆盖高层语言指令，未遥操作。真机结果属于定性示范，没有与基线真机同场对照。

## 工程部署与局限（IV-D / VI / 附录 A）

- **机体：** Unitree B2W；ZED X 双目；LiDAR-IMU 里程计与 GNSS 经 Holistic Fusion 融合；编排器 Qwen3.8-Flash-Next 运行于远端 4×H100，平均约 1.3 s/决定；trace predictor 单次约 1.5 s/次，Jetson AGX Thor；RL 控制在 Jetson AGX Orin、5 Hz。
- **异步延迟：** 新指令从采图到影响路径约 3 s（约 3 m @1 m/s）；地面过滤与低层避障兜底，不代表能对交通信号/电车进行安全关键反应。
- **请求/防护：** 编排器每 2 s 请求且最多一条在途；终点指令需足够近、过街指令防过早触发；超过 20 s 无有效指令保持停车。轨迹按新指令、行进 3 m、执行 40% 或 5 s 等条件重推。
- **失败模式：** 高层指令过早/过晚、转弯后忘记恢复直行；仅前视视野、短历史与 GNSS 长时掉线会放大问题。论文仿真不含动态交通和行人，尚不满足红灯/电车等安全关键交互。
- **开源声明：** 作者在 Introduction 说 *will release* 仿真场景、基准、数据、模型、代码；这是**未来承诺**，不是这些资源已经发布的证据。项目页本次访问失败，公开仓库与权重状态仍待核对，见 [项目页核对记录](../sites/waytrace-project.md)。

## 与 wiki 的映射

- [WayTrace 唯一实体](../../wiki/entities/paper-waytrace.md)
- [视觉–语言导航任务](../../wiki/tasks/vision-language-navigation.md)
- [具身智能小站 11-05 期技术图谱](../../wiki/overview/embodied-station-2026-11-05-43-papers-technology-map.md)

## 一手来源与二手线索分离

- **唯一主要技术依据：** [arXiv:2611.05129v1 原始全文](https://arxiv.org/html/2611.05129v1)。
- **项目主页：** [WayTrace](https://way-trace.github.io/)（论文作者提供链接，网页正文获取受阻）。
- **早期发现线索（不作实验数字依据）：** [具身智能小站 2026-11-05 综述](../blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)。
