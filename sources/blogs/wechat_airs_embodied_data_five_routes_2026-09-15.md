# wechat_airs_embodied_data_five_routes_2026-09-15

> 来源归档（blog / 微信公众号）

- **标题：** 具身智能最强教具！数据采集五大路线盘点
- **类型：** blog（产业研究 / 数采路线盘点）
- **作者：** AIRS产业研究（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/9NPuxYkfRjm3qmFuFNKiDA
- **发表日期：** 2026-09-15
- **入库日期：** 2026-09-20
- **抓取方式：** 移动端 WeChat UA + HTML 解析（`curl`；`--no-images`）
- **原始落盘：** [wechat_airs_embodied_data_five_routes_2026-09-15.md](../raw/wechat_airs_embodied_data_five_routes_2026-09-15.md)
- **一句话说明：** AIRS 产业研究盘点具身数采五大路线（真机/UMI/动捕/第一视角/仿真合成）及装备分支与代表案例；强调 2026 数采规模化与「有效数据匹配模型需求」。

## 核心摘录（归纳，非全文）

### 主判断

- **2026 = 数采规模化元年**（引述光轮智能杨海波）：WRC/WAIC 数采设备展示量显著增加。
- **五条路线并行、无绝对优劣**：质量 / 成本 / 可扩展 / 跨本体 / 真实性 trade-off；选型取决于训练目标、构型、任务精细度与成本。
- **有效数据 ≠ 录制时长**：需任务切分、标签、筛选与真机评测反向指导。

### 五条路线 × 分支 × 代表

| 路线 | 分支 | 学术锚点 | 产业/产品案例（文内） |
|------|------|----------|----------------------|
| **1 真机采集** | VR 遥操作 | ALOHA · Mobile ALOHA · Open-TeleVision | 艾欧智能 TeleXperience |
| | 主从臂 | 同上 | 松灵 Cobot Magic |
| **2 UMI 及衍生** | 手持夹爪 | UMI (RSS 2024) | 鹿明 FastUMI Pro |
| | 可穿戴手接口 | DexUMI (CoRL 2025, 86%) | — |
| **3 人体/手部动捕** | 外部光学 | DeepMimic · DexCap | NOKOV |
| | 惯性穿戴 | — | 诺亦腾 PN Studio |
| | 精细手部 | DexCap | DexCap |
| **4 第一视角** | 轻量视觉 | Ego4D · Ego-Exo4D | 自变量 QUANXTA Zero-E0 |
| | 视触融合 | — | 它石智航 SenseHub |
| **5 仿真与合成** | 程序化仿真 | MimicGen | RoboTwin 2.0 |
| | 真实场景重建 | — | 光轮智能 SimReady / Lightwheel |
| | 世界模型+动作提取 | GR00T-Dreams DreamGen | NVIDIA GR00T-Dreams |

### 各路线局限（文内）

- **真机**：规模受机器人数、人力、维护、复位效率限制；可补自主/接管/失败数据。
- **UMI 系**：无本体 ≠ 免适配；灵巧手需外骨骼与视觉编译。
- **动捕**：轨迹 alone 不够描述接触；需力触/物体状态。
- **第一视角**：视频→动作需重建/提取；与动捕/UMI 有交叉。
- **仿真/合成**：物理属性偏差或视觉合理≠可执行；WM 生成需验证接触与 IDM 精度。

## 对 wiki 的映射

- [embodied-data-collection-five-routes-landscape](../../wiki/queries/embodied-data-collection-five-routes-landscape.md)（本次升格主页面）
- [四层采集术语地图](../../wiki/concepts/embodied-data-collection-four-layers-taxonomy.md) — 视角/设备/教法/产物正交轴
- [humanoid-robot-data-collection-landscape](../../wiki/queries/humanoid-robot-data-collection-landscape.md) — Substack 六范式产业地图（互补）
- [embodied-data-collection-to-flywheel-album](../../wiki/overview/embodied-data-collection-to-flywheel-album.md) — 具身智能前沿四篇连载
- [Teleoperation](../../wiki/tasks/teleoperation.md) · [depth-embodied-data](../../roadmap/depth-embodied-data.md)

## 可信度与使用边界

- 产业研究视角 + 公开资料与企业调研；**不代表**投资/采购评价。
- 企业案例为文内选取代表，非 exhaustive 市场地图；数字以原论文/产品为准。

## 当前提炼状态

- [x] 文章基础摘要填写
- [x] 初步 wiki 页面映射确认
