# wechat_zanehub_joint_motor_topology_selection_2026-09-20

> 来源归档（blog / 微信公众号）

- **标题：** 无框力矩、空心杯、轴向磁通：谁才是人形机器人关节电机的首选？
- **类型：** blog
- **作者：** Zane Hub（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/i1rdM39LPMbgbJsoX2CNBQ
- **发表日期：** 2026-09-20
- **入库日期：** 2026-09-20
- **抓取方式：** 移动端 WeChat UA + HTML 解析（`curl`；`--no-images`）；本环境未预装 `wechat-article-for-ai`
- **原始落盘：** [wechat_zanehub_joint_motor_topology_selection_2026-09-20.md](../raw/wechat_zanehub_joint_motor_topology_selection_2026-09-20.md)
- **一句话说明：** 从人形关节真实工况（转矩/响应/热三笔账）出发，对比无框力矩、空心杯、轴向磁通三条量产路线的物理边界、供应链与按关节分级选型；强调先定减速器再定电机、峰值 vs 连续口径与 AFM 公差链。

## 核心摘录（归纳，非全文）

### 主判断

- **没有通吃方案**：低速大扭矩、小体积、快响应、省电彼此矛盾；选型是**按关节点位**匹配拓扑，而非押注单一电机类型。
- **无框力矩 = 量产中坚**：Optimus 28 执行器、宇树 G1 自研关节均以无框力矩为源；配合谐波/行星/滚柱丝杠；**减速器路线是选型的一部分**（G1 小腿用两级行星而非谐波）。
- **空心杯 = 灵巧手事实标准**：零齿槽 + 毫秒级响应；单机转矩极小，须大减速比或腱绳；Optimus 手 12 台空心杯；新一代或部分被前臂腱绳 + 微型有齿槽电机替代。
- **轴向磁通 = 跟踪而非默认**：力矩密度/轴向薄有红利，但公差链、轴向磁拉力、定子制造、成本与热四道坎；适合踝等极扁点位与直驱探索。

### 三笔基础账（文内公式）

| 账 | 要点 |
|----|------|
| **转矩** | T_out = η·i·T_motor；J_ref = i²·J_motor — 减速比↑力控透明度↓ |
| **响应** | τ_m = J_rotor·R/(Kt·Ke) — 空心杯靠小 J 压到 ms 级 |
| **热** | 封闭关节连续输出常受散热路径而非电磁设计限制 |

### 按关节分级（文内 §五）

| 关节 | 推荐拓扑 |
|------|----------|
| 髋/膝/肩/肘等大扭矩旋转 | 无框力矩 + 谐波或行星 |
| 腰/腕/踝 | 无框小型框架 + 谐波/摆线；踝可评估 AFM |
| 手指/灵巧手 | 空心杯 + 多级行星或腱绳 |
| 线性关节 | 无框高速 + 行星滚柱丝杠 |
| 直驱/QDD 探索 | 轴向磁通候选（须解公差与热） |

### 工程坑（文内 §六）

1. **先减速器后电机** — 工作点落高效区
2. **峰值 vs 连续** — 封闭关节看连续堵转温升，预留 ≥30% 裕量
3. **热设计早进场** — 导热界面/灌封/传感布局
4. **力控看双编反馈链** — 勿只盯电机参数
5. **空心杯配驱动** — 低电感须匹配电流环/PWM
6. **AFM 先算公差** — 端面跳动与轴承配置

## 对 wiki 的映射

- [humanoid-joint-motor-topology-selection](../../wiki/queries/humanoid-joint-motor-topology-selection.md)（本次升格主页面）
- [joint-module-self-development-workflow](../../wiki/concepts/joint-module-self-development-workflow.md) — 自研模组流程互补
- [humanoid-hardware-101-integrated-actuators](../../wiki/overview/humanoid-hardware-101-integrated-actuators.md) — 集成执行器语境
- [open-source-torque-motor-em-design](../../wiki/comparisons/open-source-torque-motor-em-design.md) — 轴向/径向电磁设计工具
- [motor-torque-speed-curve](../../wiki/concepts/motor-torque-speed-curve.md) — TN 连续/峰值读图
- [unitree](../../wiki/entities/unitree.md) — G1/H1 关节参数锚点

## 可信度与使用边界

- 第三方产业解读 + 公开拆解/券商口径；转矩密度、出货量与厂商参数须以 datasheet 与台架为准。
- 无单一厂商项目页需核查开源状态；Optimus/宇树参数来自公开拆解与官方文档引用。

## 当前提炼状态

- [x] 文章基础摘要填写
- [x] 初步 wiki 页面映射确认
