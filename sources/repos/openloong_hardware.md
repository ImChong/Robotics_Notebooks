# OpenLoong Hardware（青龙公版机硬件）

> 来源归档

- **标题：** OpenLoong Hardware — 青龙全尺寸通用人形机器人硬件开源
- **类型：** repo
- **来源：** 人形机器人（上海）有限公司 / 国家地方共建人形机器人创新中心（OpenLoong 社区）
- **链接（AtomGit）：** https://atomgit.com/openloong/OpenLoongHardware/tree/main
- **链接（GitHub）：** https://github.com/loongOpen/OpenLoong-Hardware （`OpenLoongHardware` 会 301 重定向至此）
- **入库日期：** 2026-05-23
- **一句话说明：** 「青龙」公版机硬件开源包：按子系统分目录的 PDF 二维图纸、总装与技术说明文档；含腰/胸/头/腿足等模块，README 给出 43 DOF、EtherCAT、感知与运动指标。
- **沉淀到 wiki：** 是 → [`wiki/entities/openloong.md`](../../wiki/entities/openloong.md)（硬件章节）

---

## 为什么值得保留

- **全尺寸公版 CAD 级资料**：公开 waist / chest / head / leg-foot 分模块 PDF，便于与动力学模型、BOM 和装配 SOP 对照。
- **规格表可直接用于选型**：README 给出尺寸、质量、DOF、速度、负载、续航与算力上限，适合写入硬件对比与 Sim2Real 边界讨论。
- **总线与控制接口线索**：明确 **EtherCAT** 作为关节实时控制总线，与软件栈 `loong_driver_sdk` 的 ECAT 驱动描述一致。

## 仓库目录结构（README 摘录）

| 目录代号 | 子系统 | 内容概要 |
|----------|--------|----------|
| **TA00-03-00** | 腰部组件 | 腰关节、侧摆/前摆电机支架、支撑转轴、补盲相机支架等 PDF |
| **TA00-04-00** | 胸腔系统 | 前胸机架、主控/电池/云盒固定、环视与雷达转接、WiFi 散热等 |
| **TA00-07-00** | 头部感知 | 视觉/听觉/触觉/嗅觉/动觉集成相关零件图（TA00-07-01-xxxx 系列） |
| **TA00-12-00-00** | 腿足系统 | 髋/膝/踝/十字轴/连杆/足端橡胶垫等 40+ 零件 PDF + 脚部总成 |
| **根目录 PDF** | 总文档 | `青龙全尺寸通用人形机器人硬件开源内容.pdf` — 开源内容技术说明 |

> 注：当前公开树以 **PDF 工程图** 为主；三维 STEP/装配体若随版本更新，应以 AtomGit/GitHub **main** 分支实际文件为准。

## 公开技术参数（README）

| 指标 | 数值 |
|------|------|
| 外形尺寸 | 2192 mm × 1850 mm × 300 mm |
| 整机质量 | 80 kg（开放原子项目页亦写约 85 kg，以所用硬件版本 README 为准） |
| 自由度 | 43 |
| 行走速度 | ≥ 5 km/h |
| 负载能力 | ≥ 20 kg |
| 续航 | ≥ 3 h |
| 控制与感知算力 | 最大约 400 TOPS |
| 关节总线 | **EtherCAT** |
| 感知 | 3D LiDAR、深度相机、环视相机等（README 级描述） |
| 末端 | 五指灵巧手（高 DOF 仿人手） |

## 与本仓库现有资料的关系

- 硬件层是 [`OpenLoong-Framework`](https://github.com/loongOpen/OpenLoong-Framework) 中 `loong_driver_sdk`（EtherCAT/485/IMU/灵巧手）与 [`OpenLoong-Dyn-Control`](https://github.com/loongOpen/OpenLoong-Dyn-Control) MuJoCo 模型的物理参照。
- 与 [`wiki/entities/asimov-v1.md`](../../wiki/entities/asimov-v1.md) 对比：Asimov 偏 **1.2 m 级 DIY + MuJoCo 同仓**；OpenLoong 青龙为 **全尺寸公版 + 工业 EtherCAT + 国家级开源生态**。

## 对 wiki 的映射

- 在 [`wiki/entities/openloong.md`](../../wiki/entities/openloong.md) 展开「具身实体」层：子系统—图纸—总线—指标表。
- 更新 [`wiki/entities/open-source-humanoid-hardware.md`](../../wiki/entities/open-source-humanoid-hardware.md) 对比表新增 OpenLoong 行。
