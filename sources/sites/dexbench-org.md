# DexBench（dexbench.org）

- **标题：** DexBench — Defining the Benchmark for Industrial Dexterity
- **类型：** site / benchmark-spec
- **URL：** <https://dexbench.org/en/>
- **规范域：** [dexbench.org](https://dexbench.org)（`/` 301 到 `/en/`）
- **镜像 / 语言：** `/en/` · `/ko/` · `/ja/`（sitemap 无 `/zh/`）
- **机构：** 瑞沃世界（RLWRLD）· 英伟达（NVIDIA）（新闻稿称合作；站点正文以任务规范为主，未单独署名作者）
- **入库日期：** 2026-08-29
- **sitemap lastmod：** 2026-06-23
- **配套产品：** 同机构 [All Hands Up](./allhandsup-org.md)（硬件档案）；[RLDX-1](../repos/rldx-1.md)（灵巧操作 VLA）
- **代码：** 截至入库日 **无独立 GitHub 仓**。`RLWRLD/DexBench`、`RLWRLD/dexbench` 均 404；`RLWRLD` 组织公开仓仅 `RLDX-1`、`openarm_description`、`ethercat_driver_ros2`。NVIDIA [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) README 将 **NVIDIA DexBench** 列在 “coming soon” 清单（与 NIST Board 1、GR00T Industrial、RoboLab 并列）

## 一句话摘要

RLWRLD 与 NVIDIA 发布的 **工业灵巧操作任务规范**：用 **物体状态复杂度（OSC）六轴** 诊断「为什么难」，用 **五种 Dexterity Regime** 规定「用什么能力吸收失败」，并给出 **18 项原子任务 / 55 个评测用例**、可采购实物清单与状态转移成功判据。它先是一份 **开放行业规格**，不是一份已开源、可一键跑的仿真排行榜。

## 开源状态（步骤 2.5，截至 2026-08-29）

| 项 | 结论 |
|----|------|
| 项目页 | **已开放**：<https://dexbench.org/en/>（任务定义、OSC、Regime、55 case、采购表、定制夹具说明） |
| 独立代码仓 | **未开源**：官方组织无 DexBench 仓；站点 HTML 未列 GitHub |
| Isaac Lab / Arena 集成 | **宣称将接入、尚未上架**：Arena README 写 NVIDIA DexBench *coming soon*；不要写成「已在 Arena 可跑」 |
| Hugging Face 演示集 | **公开检索可见数据卡**（本环境 HF API 因凭证 401 无法二次拉取）：`dexbench/single-lerobot`（单手、685 ep / 275,262 帧 / 14 任务 / 浮基 Shadow 右手 / LeRobot v2.1）、`dexbench/bimanual-lerobot`（双手、746 ep、56 维动作）；原始 teleop pickle 在门控 `dexbench/DexBench_dataset` |
| 论文 | **站点未挂 arXiv / PDF**；2026-06-09 新闻稿（PR Newswire）为发布叙事 |
| 排行榜 / 官方分数 | **项目页不发布逐任务成功率**；AHU 长文只给少数手的叙事分 |

**判定：部分开源 / 规范已公开。** 任务定义与采购清单可复现搭台；官方评测代码与 Arena 环境 **待发布**。HF 上的 LeRobot 回放是 **演示数据**，不是可替换 Arena 的评测入口。勿与 [sail-ucf/dexbench](https://github.com/sail-ucf/dexbench)（LLM 程序推理，ACL 2026）或 [DexVerse](../../wiki/entities/paper-dexverse.md)（UNC/HKU/Berkeley 仿真 bench）混名。

## 为何值得保留

- **补 AHU 留下的任务层：** [All Hands Up](./allhandsup-org.md) 只映射「硬件轴伤哪些任务」；本站才是 T00–T17 的正式定义。
- **把「灵巧」从 DoF 改写成能力包络：** Regime 与手指数解耦——两指夹爪若能做高接触复杂度任务，在该语境下比五指手更 dexterous。
- **工业可比口径：** 状态转移 + 可验证终态 + 市售物体 + breakdown curve（崩在哪一档参数，而不是单一成功率）。
- **选型层：** 落在具身评测链的 **③ 策略任务层 / 真机工业规格**，不要和 LIBERO / RoboCasa 仿真成功率榜直接比数字。

## 公开要点（编译自 `/en/` 全文，截至入库日）

### 站点结构

| 入口 | 内容 |
|------|------|
| `/en/`（及 `/ko/` `/ja/`） | 单页：问题陈述 → 18 任务卡 → 设计原则 → OSC → Regime → 逐任务 case → 物体表 → 采购清单 → 定制夹具 |
| sitemap | 仅上述三个语言落地页；`lastmod` 2026-06-23 |
| 子路径 `/en/tasks` | 404（任务都在首页锚点，无独立路由） |

### 四个设计原则

| 原则 | 含义 |
|------|------|
| State Transition | 只规定初态/终态，不规定怎么做 |
| State-Based Judgment | 成功看可验证终态与安全，不看轨迹相似度 |
| Real Objects | 用例物体可采购，给尺寸/重量/材料口径 |
| Breakdown Curves | 主产物是「可靠 → 失败」的参数崩塌曲线，不是单一 SR |

### Object State Complexity（OSC）

`OSC = ( C_geom, C_force, C_contact, C_obs, C_deform, C_dyn )`，六轴独立打分：

| 轴 | 一句话 |
|----|--------|
| `C_geom` | 合法位姿/接触构型有多窄（公差、对称、曲率） |
| `C_force` | 力/力矩包络有多窄（不足失败 vs 过力损坏） |
| `C_contact` | 接触模态种类与切换次数（点/线/面、粘滑滚卡） |
| `C_obs` | 任务关键状态有多少被遮挡或传感器看不到 |
| `C_deform` | 变形自由度与路径依赖（布、膜、线缆） |
| `C_dyn` | 状态变化有多快、时序窗口有多窄 |

站点强调：OSC 升高往往是 **容差包络 ε 变窄**，不是状态空间变大。工业里四类叠加：结构部分可观测、任务内 OSC 动态切换、多轴复合、产线节拍抬高 `C_dyn`。

### 五种 Dexterity Regime 与瓶颈规则

| Regime | 主问题 | 主导 OSC | 核心精度 |
|--------|--------|----------|----------|
| Grasp Diversity | 跨形状/材料/接近约束能否找到稳定抓 | `C_geom` + `C_contact` + 硬件 | 解空间覆盖率 |
| Spatial Precision | 亚毫米位姿/插入 | `C_geom` | 位姿 σ（mm / °） |
| Temporal Precision | 动态窗口里能否按时启动/切换/重规划 | `C_dyn` + `C_obs` | 时延 τ（ms） |
| Contact Precision | 接触后力/阻抗高频调节 | `C_force` + `C_contact` + `C_obs` | 力/阻抗 Δ（N / N·m） |
| Context Awareness | 阶段分解、失败诊断、恢复分支 | `C_deform` + `C_obs` | 阶段/分支正确率 |

规则 A–E：窄位姿包络 → Spatial；状态快于感知–估计–执行延迟 → Temporal；关键状态在力/触觉空间 → Contact；变形+遮挡+历史依赖 → Context；可行接触构型结构不足 → Grasp Diversity。

传感/数据优先级（站点表）：Grasp 用 RGB-D + 触觉阵列；Spatial 用高分辨率立体 / 轮廓仪；Temporal 用高帧率相机 + IMU；Contact 用 F/T + 触觉；Context 用多模态融合与阶段/失败标注。

评测流：`Real Object → OSC Analysis → Regime Classification → Task Design → Case Evaluation`。

### 18 任务 / 55 case（T00–T17）

站点卡片把编号写成 `T00`…`T17`（AHU 长文同口径）。下表 cases 数按首页卡片 + 正文 CASE 列表核对（合计 55）。

| ID | 任务 | Cases | 主导 OSC（卡片标签） | 主导 Regime（卡片标签） | 行业标签 |
|----|------|-------|----------------------|-------------------------|----------|
| T00 | Special Picking（环境利用抓、受限取物） | 4 | geom / contact / force | Grasp / Spatial / Contact | M/S/L |
| T01 | In-Hand Reorientation | 4 | geom / contact / obs / force | Grasp / Contact / Context | M/S/L |
| T02 | Bimanual Regrasping | 3 | geom / contact / force / dyn | Grasp / Temporal / Contact | M/L/S |
| T03 | Precision Insertion | 6 | geom / force / contact | Spatial / Contact / Context | M/L |
| T04 | Hand Fastening（徒手攻丝、限力矩拧紧） | 3 | force / contact / geom / obs | Grasp / Contact / Context | M/S |
| T05 | Constrained-Axis Manipulation | 5 | force / contact / geom / obs | Contact / Context | M/S |
| T06 | Control Interface Actuation | 4 | contact / force | Spatial / Contact | M/S/L |
| T07 | Force-Regulated Wiping | 2 | force / contact / dyn | Temporal / Contact | S/M |
| T08 | Flowable Material Control | 4 | force / contact / dyn | Temporal / Contact / Context | M/S |
| T09 | Fabric Folding | 2 | geom / contact / deform | Grasp / Context | S/L |
| T10 | Cable Winding | 2 | geom / contact / deform | Grasp / Contact / Context | S/L/M |
| T11 | Package Handling | 3 | geom / force / deform | Grasp / Spatial / Contact / Context | S/L/M |
| T12 | Selective Sorting & Binning | 1 | obs / geom / dyn / contact | Grasp / Spatial / Context | L/S/M |
| T13 | Heterogeneous Bin Packing | 2 | geom / obs / contact | Grasp / Spatial / Context | S/L |
| T14 | Box Sealing | 1 | geom / contact / deform | Grasp / Spatial / Contact / Context | M/L |
| T15 | Precision Arrangement | 3 | obs / geom / contact | Grasp / Spatial / Context | S/L |
| T16 | Tool-Use | 4 | geom / force / contact / obs | Grasp / Spatial / Contact / Context | M/S |
| T17 | Moving Object Interaction | 2 | dyn / obs / geom / contact | Grasp / Spatial / Temporal / Context | M/L |

行业：Manufacturing / Service / Logistics。

### 用例速查（物体 → 终态）

| Case | 物体 | 终态要点 |
|------|------|----------|
| 0-A…D | 名片 / M6 垫圈 / 垫圈堆 / L 箱 | 齐平薄物、小盘、杂堆抽一、双手托大箱 |
| 1-A…D | 美工刀 / iPhone 15 / 煮蛋（木蛋） / USB | 单手 180° 翻、掌心翻面、标记朝前、USB-A pinch |
| 2-A…C | 扫帚 / 胶合板 / 扁箱 | 双手换向、90° 板转、箱侧靠身 |
| 3-A…F | iPhone+托盘 / USB+定制插座 / 两片插头 / M6 螺栓·垫圈·螺母 | 亚毫米入托、定向插、同时入槽、徒手对牙 |
| 4-A…C | M6 螺栓/螺母 / 六角灯泡套件 | 徒手拧紧；灯泡以亮灭为成功 |
| 5-A…E | 定制阀架 / 桌面抽屉 / 铰链柜 | 球阀 90°、闸阀多圈、迷你球阀、抽拉、开门 |
| 6-A…D | 定制开关板 | 急停、旋钮、推子、拨杆 |
| 7-A/B | 毛巾 / 量杯 | 桌面洒水擦干；凹凸玻璃无残水 |
| 8-A…D | 电热水壶 / 木胶 / 燕麦 / 调味粉 | 300 ml 水（误差 >50 ml 失败）；周界胶线；300 ml 燕麦；漏斗装满瓶 |
| 9-A/B | 毛巾 / T 恤 | 三折到指定尺寸；正面朝上叠衣 |
| 10-A/B | 插线板线 / 独立线缆 | 绕本体；成圈直径上限 |
| 11-A…C | L 箱+刀 / 杯面套膜 / 鼠标盒 | 切胶带开四翼、去膜、塞回并合盖 |
| 12-A | 10 物混装 L 箱 | 按类分到桌面 |
| 13-A/B | 7 盒 / 12-A 的 10 物 | 装回 L 箱不超出沿口 |
| 14-A | L 箱 + 胶带枪 | 顶缝连续贴、切口干净 |
| 15-A…C | 三盒 / 两杯面 | 齐边等距；叠杯对 logo；并排对 logo |
| 16-A…D | 4 mm 内六角 / 电动批 / 手动批 / 10 mm 扳手 | 工具必须用上；打滑重试上限 |
| 17-A/B | 传送带上的鼠标/盒；午餐盒格 + 鸡块/西兰花样品 | 分箱；运动中入格 |

失败条件共性：禁止「外物辅助」、禁止过力硬塞、多数插入/紧固禁止改用工具（T16 相反：禁止徒手代替工具）。若干 case 串成序列（3-E → 4-A，3 螺母对牙 → 4-B，12-A → 13-B）。

### 采购与定制夹具

站点给 **约 50 个用例物体** 与 **45 条采购行**（Amazon / 五金 / 食品样品），以及五套定制件：Outlet Kit、Hole Kit（3D 打印 M6 螺母座）、Hexagonal Bulb Kit、Valve Kit、Switch Kit。评测只考机械动作（插座套件明确「不接线」）。

### Hugging Face 数据卡摘要（检索，待本机 token 复核）

公开索引的 `dexbench/single-lerobot` 写明：Isaac Lab 回放遥操作 pickle，再 `scripts/create_demo_files.py` + `scripts/convert_to_lerobot.py`（h264 CRF 18）打成 LeRobot v2.1；第三人称 + 腕部 RGB 256×256 @ 30 fps + 本体 + 关节动作。任务名形如 `Dexbench-OpenFaucet-v0`、`Dexbench-InsertPeg-v0`、`Dexbench-PlugCharger-v0` 等 **14** 项——与官网 T00–T17 **不是一一同名**，读数据时按任务名对齐，不要按编号硬套。双手子集另见 `dexbench/bimanual-lerobot`。原始目录在门控 `dexbench/DexBench_dataset` 的 `teleop_dataset/{rigid, articulation, dexterous, grasping}/`。

**注意：** 数据卡提到的 `scripts/*.py` 暗示存在内部转换栈；截至入库日 **没有** 对应的公开 GitHub 仓可核对这些脚本。当作「数据卡叙事」，不要写成可 `pip install` 的官方评测包。

### 不要混名的同名项目

| 名称 | 是什么 | 关系 |
|------|--------|------|
| 本站 DexBench | RLWRLD × NVIDIA 工业灵巧任务规范 | 本次 ingest 对象 |
| NVIDIA DexBench（Arena 文案） | Isaac Lab-Arena 即将接入的同名 bench | **同一产品线、仿真侧未上架** |
| [DexVerse](https://ycyao216.github.io/DexVerse.site/) | UNC/HKU/Berkeley 100 任务 Isaac Lab bench | **不同项目** |
| [sail-ucf/dexbench](https://github.com/sail-ucf/dexbench) | LLM 程序执行正反向推理 | **完全无关** |
| DexCompose 仓内 `source/dexbench` | 第三方 Isaac Lab 扩展包名 | **不要当成官方仓** |

## 关联资料

- 同机构硬件档案：[All Hands Up](./allhandsup-org.md)
- 同机构 VLA：[RLDX-1](../repos/rldx-1.md)
- 发布新闻（2026-06-09）：[PR Newswire](https://www.prnewswire.com/news-releases/rlwrld-launches-dexbench-initiative-to-define-next-generation-industry-standards-for-humanoid-ai-in-collaboration-with-nvidia-302795350.html)
- Arena 生态（DexBench 仍标 coming soon）：<https://github.com/isaac-sim/IsaacLab-Arena>

## 对 wiki 的映射

- [DexBench](../../wiki/entities/dexbench.md) — 规范实体、OSC/Regime 读法、开源边界
- 交叉：[All Hands Up](../../wiki/entities/all-hands-up.md)、[RLDX-1](../../wiki/entities/rldx-1.md)、[Isaac Lab](../../wiki/entities/isaac-lab.md)、[DexVerse](../../wiki/entities/paper-dexverse.md)、[具身评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)、[Manipulation](../../wiki/tasks/manipulation.md)
