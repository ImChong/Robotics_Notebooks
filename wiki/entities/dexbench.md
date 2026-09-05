---
type: entity
tags:
  - benchmark
  - dexterous
  - manipulation
  - industrial
  - isaac-lab
  - lerobot
  - rlwrld
  - nvidia
status: complete
updated: 2026-09-05
related:
  - ./all-hands-up.md
  - ./rldx-1.md
  - ./isaac-lab.md
  - ./paper-dexverse.md
  - ./paper-dexholdem.md
  - ./lerobot.md
  - ./hrdexdb-dataset.md
  - ../tasks/manipulation.md
  - ../concepts/contact-rich-manipulation.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../overview/hub-embodied-eval-benchmark.md
  - ../queries/dexterous-data-collection-guide.md
sources:
  - ../../sources/sites/dexbench-org.md
  - ../../sources/sites/allhandsup-org.md
  - ../../sources/repos/rldx-1.md
summary: "DexBench（RLWRLD × NVIDIA）：工业灵巧操作开放规格——OSC 六轴诊断难度、五种 Dexterity Regime 规定能力包络，18 任务 / 55 用例 + 可采购实物。规范已公开；官方评测仓与 Isaac Lab-Arena 集成仍标 coming soon。"
---

# DexBench（工业灵巧操作基准）

## 一句话定义

**DexBench** 是瑞沃世界（RLWRLD）与英伟达（NVIDIA）发布的 **工业灵巧操作任务规格**：用 **物体状态复杂度（OSC）** 回答「为什么难」，用 **五种 Dexterity Regime** 回答「用什么能力吸收失败」，并给出 **18 项原子任务、55 个评测用例** 与可采购实物清单。它先是一份 **可复现的行业词汇表**，不是一份已开源、可一键跑的仿真排行榜。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OSC | Object State Complexity | 六轴难度向量：几何 / 力 / 接触 / 观测 / 变形 / 动态 |
| Regime | Dexterity Regime | 五种能力包络：抓取多样性、空间/时间/接触精度、情境意识 |
| T00–T17 | DexBench atomic tasks | 18 项工业原子任务编号（AHU 长文同口径） |
| Arena | Isaac Lab-Arena | NVIDIA 统一评测核；本 bench 在 README 仍标 coming soon |
| SR | Success Rate | 单一成功率；本规格主张改看 breakdown curve |
| LeRobot | LeRobot dataset format | HF 上回放演示的 v2.1 封装，不是官方评测入口 |

## 为什么重要

- **补「硬件档案」缺的任务层。** [All Hands Up](./all-hands-up.md) 只映射「哪些硬件轴伤哪些任务」；T00–T17 的正式定义在本站。选型时先看 OSC 崩在哪一轴，再回头看 Kapandji / 背驱 / 指垫。
- **把「灵巧」从 DoF 改写成能力包络。** 站点明确：灵巧是 **解** 的属性，不是手的属性。两指夹爪若能完成高接触复杂度任务，在该语境下比五指手更 dexterous。这和 [AHU](./all-hands-up.md)「规格表读不出任务成败」同一判断，只是评测对象从硬件换成任务。
- **工业可比口径，而不是实验室同名不同义。** 物流的 pick 和电子装配的 pick 共享单词、不共享方法。DexBench 用状态转移 + 可验证终态 + 市售物体，强迫不同实验室比同一套初态/终态/失败条件。
- **落在评测链的 ③ 层，但不是 LIBERO 式仿真 SR 榜。** 见 [具身评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md)：本页是 **真机工业规格 + 诊断语言**；在 Arena 上架之前，不要拿它和 [DexVerse](./paper-dexverse.md) / RoboCasa 的数字直接比。

## 核心原理

### 两维分类：为什么难 × 用什么吸收

```mermaid
flowchart LR
  obj[市售实物] --> osc["OSC 六轴诊断\nC_geom … C_dyn"]
  osc --> reg["Regime 分类\n五种能力包络"]
  reg --> task[18 原子任务设计]
  task --> cases[55 用例评测]
  osc -.->|规则 A–E| reg
```

每个任务同时落在两条独立轴上：

1. **OSC 诊断失败从哪来** — `OSC = (C_geom, C_force, C_contact, C_obs, C_deform, C_dyn)`。升高通常是 **容差包络 ε 变窄**，不是状态空间变大。同一句「插入」：±2 mm trivial，±0.2 mm 未解。
2. **Regime 规定用什么能力吸收** — Grasp Diversity / Spatial / Temporal / Contact Precision / Context Awareness。一任务可同时要求多 Regime；组合才是真实难度。

操作被写成 **部分可观测的混合动力学**：接触模态（粘 / 滑 / 滚 / 卡 / 脱）制造离散切换；内部接触与摩擦对视觉结构不可见；动态窗口里端到端延迟直接决定成败。因此「灵巧」定义为：**能观测哪些状态、能稳住哪些模态切换**，而不是「能动多少关节」。

### OSC 六轴（读失败用）

| 轴 | 问什么 | 工业直觉 |
|----|--------|----------|
| `C_geom` | 合法位姿/接触有多窄 | USB 反插、螺栓对牙、托盘亚毫米间隙 |
| `C_force` | 力/力矩包络有多窄 | 蛋壳 vs 拧紧；不足失败、过力损坏 |
| `C_contact` | 接触模态种类与切换次数 | 指步态、缠绕、胶带切开 |
| `C_obs` | 关键状态有多少看不见 | 螺纹啮合、膜下缝、孔内批头 |
| `C_deform` | 变形自由度与路径依赖 | 布、膜、线缆、粉体架桥 |
| `C_dyn` | 状态变化有多快 | 倾倒过冲、传送带窗口、擦拭节拍 |

工业四条叠加：结构部分可观测（几乎每项都有 `C_obs`）、单次执行内 OSC 动态切换（抓低 → 对准高 → 放置低）、多轴复合、产线节拍抬高 `C_dyn`。

### 五种 Regime 与瓶颈规则

| Regime | 主问题 | 主导 OSC | 看什么精度 |
|--------|--------|----------|------------|
| Grasp Diversity | 跨形状/材料/接近约束能否抓住 | geom + contact + 硬件 | 解空间覆盖 |
| Spatial Precision | 紧公差下的位姿/插入 | geom | mm / ° |
| Temporal Precision | 动态窗口里的启动/切换/重规划 | dyn + obs | ms 级时延 |
| Contact Precision | 接触后力/阻抗调节 | force + contact + obs | N / N·m |
| Context Awareness | 阶段、失败诊断、恢复分支 | deform + obs | 阶段/分支正确率 |

规则 A–E：窄位姿包络 → Spatial；状态快于感知–估计–执行环 → Temporal；关键状态在力/触觉空间 → Contact；变形 + 遮挡 + 历史依赖 → Context；可行接触构型结构不足 → Grasp Diversity。

传感/采数优先级随 Regime 变：Spatial 要位姿真值，Contact 要力/阻抗轨迹，Context 要阶段与失败分类标注——和 [灵巧操作数据采集指南](../queries/dexterous-data-collection-guide.md) 的「先定测什么再定采什么」一致。

### 四个设计原则

| 原则 | 工程含义 |
|------|----------|
| 状态转移 | 只写初态/终态，方法开放；禁止把示范轨迹当成功条件 |
| 状态判据 | 终态可验证 + 安全未破；轨迹像不像人不计分 |
| 真实物体 | 可采购、给规格；定制件只补夹具，不发明不可买的「基准物」 |
| Breakdown curve | 主产物是「可靠 → 失败」的参数崩塌曲线，不是单一 SR |

### 18 任务速查（T00–T17）

完整 case 与采购行见 [站点归档](../../sources/sites/dexbench-org.md)。下表只保留选型时要用的编号与能力焦点。

| ID | 任务 | n | 能力焦点 |
|----|------|---|----------|
| T00 | Special Picking | 4 | 齐平薄物、杂堆抽一、双手托超手展箱 |
| T01 | In-Hand Reorientation | 4 | 单手翻刀/手机/蛋/USB；禁止双手 |
| T02 | Bimanual Regrasping | 3 | 长杆换向、大板 90°、箱侧靠身 |
| T03 | Precision Insertion | 6 | 入托、USB、两片插头、M6 对牙 |
| T04 | Hand Fastening | 3 | 徒手拧螺栓/螺母；灯泡以亮灭为成功 |
| T05 | Constrained-Axis | 5 | 阀、抽屉、铰链门：只许沿约束轴用力 |
| T06 | Control Interface | 4 | 急停 / 旋钮 / 推子 / 拨杆 |
| T07 | Force-Regulated Wiping | 2 | 桌面与凹凸玻璃均匀压力 |
| T08 | Flowable Material | 4 | 水 / 木胶 / 燕麦 / 粉末计量 |
| T09 | Fabric Folding | 2 | 毛巾三折、T 恤整形 |
| T10 | Cable Winding | 2 | 绕插线板本体 vs 自由成圈 |
| T11 | Package Handling | 3 | 切胶带、去膜、插回纸盒 |
| T12 | Selective Sorting | 1 | 10 物 5 类混箱分类 |
| T13 | Heterogeneous Packing | 2 | 规则盒 vs 12-A 的 10 物回装 |
| T14 | Box Sealing | 1 | 胶带枪连续贴顶缝 |
| T15 | Precision Arrangement | 3 | 齐边等距、叠杯/并排对 logo |
| T16 | Tool-Use | 4 | 内六角 / 电批 / 手批 / 扳手；禁止改徒手 |
| T17 | Moving Object | 2 | 传送带分拣与运动中入格 |

行业标签覆盖制造 / 服务 / 物流。若干 case 串成序列（例如 3-E → 4-A，12-A → 13-B），读分时不要把下游任务当成独立冷启动。

## 工程实践

| 步骤 | 做法 |
|------|------|
| 读规格 | 打开 [dexbench.org/en/](https://dexbench.org/en/)；韩/日镜像 `/ko/` `/ja/`。无中文站 |
| 先画 OSC 再选任务 | 不要从任务名出发。先标六轴，再套规则 A–E 看缺哪条 Regime |
| 搭真机台 | 按站点 Purchase List 买市售物；五套定制夹具（插座 / M6 孔座 / 灯泡 / 阀架 / 开关板）只补固定，不接线、不通水 |
| 判成功 | 对照每个 CASE 的 Initial / Goal / Failure；禁止「外物辅助」、禁止过力硬塞 |
| 对硬件 | 回 [AHU](./all-hands-up.md) 看拇指 ROM、DIP、最小直径、背驱是否伤该 Regime；**不要**在 AHU 找 T00–T17 分数 |
| 对模型 | [RLDX-1](./rldx-1.md) 公开数字在 LIBERO / RoboCasa / GR-1，**不是** DexBench 官方分 |
| 仿真 | Isaac Lab-Arena README 仍写 NVIDIA DexBench *coming soon*。在官方包出现前，不要假设 `list_envs.py` 能列出 `Dexbench-*-v0` |
| 演示数据 | 公开检索可见 `dexbench/single-lerobot`（单手 Shadow 回放，14 任务）与 `dexbench/bimanual-lerobot`；原始 pickle 在门控 `dexbench/DexBench_dataset`。任务名与 T00–T17 **不是一一同名** |
| 源码运行时序图 | **不适用**：截至 2026-08-29 无官方可运行评测仓；数据卡提到的 `scripts/convert_to_lerobot.py` 未随公开 GitHub 发布 |

**同名消歧：**

| 名字 | 是什么 |
|------|--------|
| 本页 DexBench | RLWRLD × NVIDIA 工业任务规格 |
| Arena 文案里的 NVIDIA DexBench | 同一产品线的仿真侧，**未上架** |
| [DexVerse](./paper-dexverse.md) | UNC/HKU/Berkeley 的 100 任务 Isaac Lab bench |
| [sail-ucf/dexbench](https://github.com/sail-ucf/dexbench) | LLM 程序推理，无关 |
| 第三方仓里的 `source/dexbench` | 例如 DexCompose 的本地扩展包名，不是官方仓 |

## 局限与风险

- **开源边界：** 规范页 **已开放**；截至 2026-08-29 **无官方 GitHub**。`RLWRLD` 组织公开仓不含 DexBench。[Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) 将其列为 coming soon。不要写成「已开源评测栈」或「已在 Arena 可跑」。
- **HF 数据 ≠ 官方评测。** LeRobot 回放是演示；数据卡脚本路径无法从公开仓核对。本环境 HF API 曾 401，下载前应自己登录复核门控与许可。
- **没有官方分数榜。** 站点不发布逐任务 SR；AHU 对 SharpaWave / DG-5F-S 的「DexBench 表现」是操作叙事，不是中立排行。
- **18 ≠ 工业全集。** 用例偏向桌面/台架可采购物（iPhone 15、杯面、木蛋）。没有全身 loco-manipulation，也没有产线节拍下的多机协作。
- **Breakdown curve 尚未作为可下载产物。** 原则写了「看崩塌曲线」，页上给出的是离散 case，不是连续参数扫描数据。
- **与仿真灵巧 bench 不可直接比。** [DexVerse](./paper-dexverse.md) 已开源 Isaac Lab 任务与示范；本页是真机规格。同一词「插入」两边的物体、公差、成功谓词都不同。
- **发布叙事含机构模型。** 2026-06-09 新闻稿用 [RLDX-1](./rldx-1.md) 在既有仿真 bench 上的结果为合作背书；那是另一套数字，不要读成本 bench 基线。

## 关联页面

- [All Hands Up](./all-hands-up.md) — 同机构硬件层；本页是它显式留空的任务层
- [RLDX-1](./rldx-1.md) — 同机构灵巧 VLA；公开评测不在 DexBench 上
- [Isaac Lab](./isaac-lab.md) — Arena 计划接入处；当前仍是 coming soon
- [DexVerse](./paper-dexverse.md) — 易混名的开源仿真灵巧 bench
- [DexHoldem](./paper-dexholdem.md) — 易混名的真机扑克协议；分数已出，对象是牌/筹码不是工业 OSC
- [LeRobot](./lerobot.md) — HF 回放集的封装格式
- [HRDexDB](./hrdexdb-dataset.md) — 同机构人–机配对抓取数据，不是本任务套件
- [Manipulation](../tasks/manipulation.md) — 操作任务总览
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md) — T03–T08 / T16 的接触力主线
- [具身评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — ③ 层：工业规格，不是仿真 SR 榜
- [具身评测基准枢纽](../overview/hub-embodied-eval-benchmark.md) — 四层入口
- [灵巧操作数据采集指南](../queries/dexterous-data-collection-guide.md) — Regime → 传感/标注优先级

## 参考来源

- [DexBench 站点归档](../../sources/sites/dexbench-org.md)
- [All Hands Up 站点归档](../../sources/sites/allhandsup-org.md) — 硬件轴与「任务分在独立站」的原分工
- [RLDX-1 仓库归档](../../sources/repos/rldx-1.md)
- 项目页：<https://dexbench.org/en/>（2026-08-29 拉取；sitemap `lastmod` 2026-06-23）

## 推荐继续阅读

- 英文规范页：<https://dexbench.org/en/>
- Isaac Lab-Arena（DexBench 仍标 coming soon）：<https://github.com/isaac-sim/IsaacLab-Arena>
- 发布说明（2026-06-09）：[PR Newswire](https://www.prnewswire.com/news-releases/rlwrld-launches-dexbench-initiative-to-define-next-generation-industry-standards-for-humanoid-ai-in-collaboration-with-nvidia-302795350.html)
- 对照开源仿真 bench：[DexVerse 项目页](https://ycyao216.github.io/DexVerse.site/)
