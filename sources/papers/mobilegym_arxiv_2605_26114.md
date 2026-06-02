# MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research

> 来源归档（ingest）

- **标题：** MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research
- **类型：** paper
- **来源：** arXiv abs / HTML；项目页、官网与 GitHub README 交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2605.26114>
  - <https://arxiv.org/pdf/2605.26114>
  - <https://mobilegym.github.io/>
  - <https://mobilegym.dev>
- **作者：** Dingbang Wu*, Rui Hao*, Haiyang Wang, Shuzhe Wu, Han Xiao, Zhenghong Li, Bojiang Zhou, Zheng Ju, Zichen Liu, Lue Fan‡, Zhaoxiang Zhang†（* 共同一作；† 通讯；‡ 项目负责人；中科院自动化所、北大、港中文等）
- **入库日期：** 2026-06-02
- **一句话说明：** 浏览器托管的 Android 式日常 App 仿真：全环境 **结构化 JSON 状态** 支持确定性裁判、毫秒级快照分叉与单机数百并行实例；**MobileGym-Bench** 含 416 参数化任务模板（256 test + 160 train），GRPO 在仿真侧 +12.8 pt SR，真机信号子集保留 **95.1%** 训练增益。

## 核心论文摘录（MVP）

### 1) 问题：真机 / 模拟器路线的结构性盲区

- **链接：** <https://arxiv.org/abs/2605.26114> §1
- **摘录要点：** 移动 GUI Agent 在截图 + 自然语言指令上进展很快，但评测与训练环境分裂：**模拟器**（AndroidWorld、AndroidLab）可重复却多限于系统工具与开源 App，大规模在线 RL 需重量级模拟器实例；**真机**（MobileBench-OL）覆盖日常 App，但账号、后端状态、版本漂移与真钱/真消息后果使 episode 难控制、难复现、难并行。日常 App 对 Agent 而言 **不可读**（余额/订单在加密 DB 与云端）、**不可写**（难重置到已知初态）、**不可分叉**（GRPO 需同初态多 rollout）、且许多动作 **不可逆**。
- **对 wiki 的映射：**
  - [MobileGym](../../wiki/entities/mobilegym.md) — 三条墙与「交互保真、不必复刻专有后端」的设计原则。

### 2) 平台：分层状态 + 声明式导航 + 浏览器栈

- **链接：** <https://arxiv.org/abs/2605.26114> §3；[GitHub README](https://github.com/Purewhiter/mobilegym)
- **摘录要点：**
  - **目标：** 交互保真（屏幕、触控、跨 App、任务相关状态转移），非像素级 Android 内部或真实后端。
  - **分层状态：** World Data（只读公共实体）+ Runtime Overlay（Agent 可写）+ OS Runtime；仅 runtime 暴露给配置/reset/judge/diff。
  - **声明式导航：** 每 App UI 为开发期构建的有限状态机 spec，驱动运行时导航与任务轨迹枚举。
  - **三层栈：** OS（TaskManager、Intent、通知等）→ Apps（28 个：12 日常 + 16 系统）→ Benchmark（`bench_env/`，Playwright，17 动作抽象，坐标归一化 [0,1000]）。
  - **资源：** 单实例约 **400 MB RAM**、**50 MB 磁盘**、**~3 s** 冷启动；单机可 host **数百** 并行实例。
- **对 wiki 的映射：**
  - [MobileGym](../../wiki/entities/mobilegym.md) — 架构 Mermaid、扩展契约（manifest 自动发现）。

### 3) 可验证信号：状态裁判 + AnswerSheet + 全环境副作用检测

- **链接：** <https://arxiv.org/abs/2605.26114> §3.2, §4.2
- **摘录要点：**
  - 每任务 **确定性 judge** 检查结构化状态，亚毫秒级，无需 VLM 判分。
  - **全环境 state diff** 检测任务外意外变更（USE 指标）；真机 / 截图裁判无法可靠发现 App 内部副作用。
  - **AnswerSheet：** 查询类任务通过 GUI 表单提交 **类型化字段**，避免自由文本匹配与 CoT 泄露假阳性。
- **对 wiki 的映射：**
  - [MobileGym](../../wiki/entities/mobilegym.md) — 评测指标表（SR/PR/FC/USE/OT）。

### 4) MobileGym-Bench 与实验

- **链接：** <https://arxiv.org/abs/2605.26114> §4–5；<https://mobilegym.dev>
- **摘录要点：**
  - **416** 参数化模板（**256 test + 160 train** 严格不相交），**28** App；运行时通过指令变体、参数采样、环境注入组合出 **>27k** 实例（连续参数更多）。
  - **Taxonomy 四轴：** Scope（S1–S3 App 数）、Objective（operate/query/hybrid）、Composition（atomic/sequential/transfer/deep-dive）、Difficulty（L1–L4，八模型后验标定）。
  - **主结果：** 9 个 Agent 在 test 集 SR **9.4%–58.8%**（Gemini 3.1 Pro 最高）；VLM 判真机轨迹 **10.2%** 误判 vs 程序化裁判 **0** 假接受/拒绝（发布校验集）。
  - **GRPO：** Qwen3-VL-4B，10 步，3×RTX Pro 6000 + **96** 并行浏览器；仿真 test SR **9.4%→22.2%**（+12.8 pt）；59 任务真机可跑信号子集 **+42.8 pt sim → +40.7 pt real**（**95.1%** 保留）。
- **对 wiki 的映射：**
  - [MobileGym](../../wiki/entities/mobilegym.md) — 与 AndroidWorld / MobileBench-OL 对照表、Sim-to-Real 摘要。

### 5) 公开资源与许可

- **代码：** <https://github.com/Purewhiter/mobilegym>（Apache-2.0）
- **数据：** `mobilegym-data` 发布包（CC BY-NC 4.0，~1.4 GB 合成内容）
- **依赖：** Node ≥ 22，Python ≥ 3.11，Playwright Chromium
- **对 wiki 的映射：**
  - [purewhiter_mobilegym 仓库归档](../repos/purewhiter_mobilegym.md)

## 当前提炼状态

- [x] arXiv 摘要、Table 1 环境对比、§3–5 实验与官网 TL;DR 已摘录
- [x] GitHub README Quick Start、架构与 Agent 适配器列表已对齐
- [x] wiki 映射：`wiki/entities/mobilegym.md`；交叉链接在 wiki 实体页维护
