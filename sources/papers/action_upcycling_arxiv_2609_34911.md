# Action Upcycling: Don't Throw Away the Tail（arXiv:2609.34911）

> 来源归档（ingest）

- **标题：** Don't Throw Away the Tail: Action Upcycling for Policy Acceleration
- **类型：** paper / vla / action-chunking / deployment / test-time
- **arXiv abs：** <https://arxiv.org/abs/2609.34911>
- **PDF：** <https://arxiv.org/pdf/2609.34911>
- **项目页：** <https://acupcycling.github.io/> — 归档见 [`sources/sites/acupcycling-github-io.md`](../sites/acupcycling-github-io.md)
- **代码：** **已开源** — <https://github.com/star-kwon/action-upcycling>（Apache-2.0，OpenPI 端口 + LIBERO 评测脚本）；归档见 [`sources/repos/action-upcycling.md`](../repos/action-upcycling.md)
- **机构：** 成均馆大学（Sungkyunkwan University）、韩国科学技术院（KAIST）— Taesung Kwon、Jangho Park 等；Jong Chul Ye（KAIST）
- **入库日期：** 2026-09-30
- **一句话说明：** **训练-free** 部署规则：在 chunk 尾部 **动作速度平滑** 时继续执行被丢弃的 tail，用累积速度波动 \(c_k\) 与阈值 \(\tau\) 自适应拉长 execution horizon；四策略 × 三基准 **policy call 减 1.2–1.7×** 且 **11/11 单元格成功率 ≥ baseline**。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://acupcycling.github.io/> | 主表、与 AAC / AutoHorizon 对比、YAM 真机 |
| GitHub | <https://github.com/star-kwon/action-upcycling> | `examples/libero/run_upcycling.sh`、OpenPI policy server |
| 对照 | AutoHorizon（attention）、AAC（多样本） | 本文只读 **已采样 chunk**，无额外前向 |

## 摘要级要点

- **问题：** Chunk 策略每轮预测 \(H\) 步、只执行前缀 \(h\)、丢弃 tail；短 \(h\) 反应快但 **policy call 密**；自适应 horizon 方法多依赖 **模型内部 attention** 或 **多次采样**。
- **观察：** 被丢弃动作与 replan 版本 **强相关**（Pearson 0.89–0.98）；RMSE 随 tail **速度波动** 上升。
- **方法：** 从 \(h+1\) 起累积 \(\|v_j-v_{j-1}\|_2\) 得 \(c_k\)；当 \(c_k\le\tau\) 继续执行 tail；\(\tau\) 由目标 upcycling ratio \(r=\bar h/h\) 在 **历史 chunk 信号池** 上离线搜索（每模型×基准一个标量）。
- **评测：** π0.5、SmolVLA、GR00T N1.7、FastWAM；LIBERO、LIBERO-Plus、RoboTwin 2.0；YAM 真机 π0.5 / GR00T N1.6。

## 核心摘录（面向 wiki 编译）

### 1) LIBERO 主表（节选）

| Model | Baseline SR | + Up cycling SR | calls/ep（baseline → ours） |
|-------|-------------|-----------------|------------------------------|
| π0.5 | 96.9 | **97.9** | 32.4 → **21.9**（1.5×） |
| SmolVLA | 82.6 | 82.8 | 19.3 → 14.7 |
| GR00T N1.7 | 96.1 | 96.6 | 22.6 → 15.0 |
| FastWAM | 97.4 | 97.6 | 15.7 → 13.4 |

### 2) π0.5 vs 自适应 horizon（LIBERO）

| Method | succ. | calls/ep | s/ep |
|--------|-------|----------|------|
| Baseline | 96.9 | 32.4 | 4.41 |
| AAC (K=20) | 97.3 | 23.5 | **18.9**（单 call 延迟 6–7×） |
| AutoHorizon | 97.4 | 22.3 | 3.03 |
| Action Upcycling | **97.9** | **21.9** | **3.00** |

### 3) 真机 YAM（π0.5，汇总）

- 成功率 **72/80 → 77/80**；calls/ep **48.3 → 34.0**（1.42×）；时间/ep **24.9 s → 21.5 s**。

## 对 wiki 的映射

- 新建：[paper-action-upcycling](../../wiki/entities/paper-action-upcycling.md)
- 交叉：[action-chunking](../../wiki/methods/action-chunking.md)、[vla](../../wiki/methods/vla.md)、[paper-autohorizon](../../wiki/entities/paper-autohorizon.md)、[paper-flashvla](../../wiki/entities/paper-flashvla.md)

## 当前提炼状态

- [x] arXiv + 项目页 + GitHub 核查（2026-09-30）
- [x] 开源：Apache-2.0 评测脚本 + OpenPI 集成
- [x] 源码运行时序图（LIBERO baseline vs upcycling 脚本）
