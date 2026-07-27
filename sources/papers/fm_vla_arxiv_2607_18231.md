# FM-VLA: Force-based Memory for Vision-Language-Action Models in Contact-Rich Manipulation

> 来源归档（ingest）

- **标题：** FM-VLA: Force-based Memory for Vision-Language-Action Models in Contact-Rich Manipulation
- **类型：** paper
- **来源：** arXiv abs / PDF；项目页与 GitHub 占位仓交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2607.18231>
  - <https://ar5iv.labs.arxiv.org/html/2607.18231>
  - <https://qft-333.github.io/FM-VLA-Page/>
  - <https://github.com/qft-333/FM-VLA>
- **作者：** Ruicheng Li, Qixiu Li, Ruichun Ma, Yu Deng, Lin Luo, Zhiying Du, Jianfeng Xiang, Huizhi Liang, Ruicheng Wang, Jiaolong Yang, Baining Guo
- **机构：** 清华大学（Tsinghua）、微软研究院（Microsoft Research）、复旦大学（Fudan）、中国科学技术大学（USTC）；† 微软研究院实习期间完成；* 通讯 Jiaolong Yang
- **入库日期：** 2026-07-27
- **一句话说明：** 在 **π₀.₅** 上为接触丰富、非马尔可夫操作引入 **力觉（wrench）长程记忆**：用 **Perceiver-IO VAE** 把整集腕部六轴力/力矩压缩成 **K=8** 力记忆 token，并附 **~0.9 s** 短窗关节状态 token，注入 flow-matching action expert；智元 **G1 双臂** 三项记忆依赖任务平均成功率 **83.3%**，相对视觉记忆 **π-MEM** 推理仅 **+3.3 ms**。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| 项目页 | <https://qft-333.github.io/FM-VLA-Page/> — 方法/任务视频/结果表；头部 **Code** 链到 GitHub |
| GitHub | <https://github.com/qft-333/FM-VLA> — 公开仓，含 demo mp4/gif 与 README |
| 可运行代码 / 权重 | **否** — README：「Code will be released soon.」 |
| 结论 | **宣称将开源 / 占位仓**（截至入库日无可运行训练/推理入口） |

## 核心论文摘录（MVP）

### 1) 问题：视觉记忆在接触事件上失效

- **链接：** <https://arxiv.org/abs/2607.18231> §1
- **摘录要点：** 多数 VLA 仍是 **Markovian** $\pi(a_t\mid o_t,l)$；记忆增强多走 **视觉/语言**（MemoryVLA、MEM）。当重复按键行程极小、擦拭/杯子复位后画面几乎不变时，视觉历史 **昂贵且不可靠**。ForceVLA / TA-VLA 等把力当作 **短窗即时条件**，可改善接触细节，但 **不累积**「已按几次 / 已擦几轮」的长程事件计数。
- **对 wiki 的映射：**
  - [FM-VLA（论文实体）](../../wiki/entities/paper-fm-vla.md) — 问题定位。
  - [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md) — 接触过程本身作为任务状态。
  - [VLA](../../wiki/methods/vla.md) — 记忆增强子路线对照。

### 2) 方法：Force-VAE + 短状态窗 → action expert 后缀

- **链接：** arXiv §3；Appendix E
- **摘录要点：**
  - **骨干：** OpenPI **π₀.₅**（PaliGemma + SigLIP + flow-matching action expert）；先在智元 Challenge 数据上再预训练，再于本任务 demo 后训练。
  - **长程 wrench：** 右腕 **6 轴 F/T**（传感器 100 Hz，策略侧下采 **30 Hz**）；一阶 EMA $\alpha=0.3$；训练时随机噪声前缀（最多 ~10 s）防止用 **序列长度** 捷径猜进度。
  - **Force-VAE：** Perceiver-IO 编解码；$K=8$ latent；$d_z=96$；masked reconstruction + free-bits KL；Stage 1 任务无关重建预训练后 **冻结**，推理只用后验均值。
  - **短状态窗：** 最近约 **0.9 s** 关节+夹爪（$10\times16$）线性投影为 **1 token**，缓解「接触前重复动作」。
  - **注入：** `[noisy actions ‖ Z_f (8) ‖ z_s (1)]` 挂在 action expert **后缀**，保留预训练 RoPE 位置。
- **对 wiki 的映射：**
  - [FM-VLA](../../wiki/entities/paper-fm-vla.md) — 流程总览与工程表。
  - [π₀.₇ Policy](../../wiki/methods/pi07-policy.md) / [π₀ Policy](../../wiki/methods/π0-policy.md) — π 系骨干。

### 3) 真机三项任务与主结果

- **链接：** arXiv §4；项目页
- **摘录要点：**
  - **平台：** 智元 **AgiBot G1** 双臂（7+7 DoF + 双夹爪）；头+双腕 RGB；VR 遥操作采数（Cups 200 / Buttons 350 / Wipe 200 demos）。
  - **任务：** (1) 两杯下找隐藏木块（复位后外观还原）；(2) 按蓝键恰好 $N\in\{1,2,3\}$ 次（行程几乎不可见）；(3) 海绵擦碗恰好 $N$ 轮。
  - **基线：** π₀.₅（无历史）、TA-VLA（短窗力）、π-MEM（MEM 式视觉记忆重实现于 π₀.₅）。
  - **成功率（18 trials/任务）：** FM-VLA **100 / 72.2 / 77.8** → 平均 **83.3%**；π₀.₅ **27.8%**；TA-VLA **22.2%**；π-MEM **53.7%**。
  - **延迟（RTX 4090）：** FM-VLA **64.0 ms（+3.3）** vs π-MEM K=5 **+39.1** / K=16 **+129.3**。
- **对 wiki 的映射：**
  - [FM-VLA](../../wiki/entities/paper-fm-vla.md) — 结果表。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 记忆依赖接触操作代表工作。

### 4) 消融：模态互补与 VAE 压缩

- **链接：** arXiv §4.3
- **摘录要点：** 仅力历史平均 **25.9%**（接触前运动失控）；仅状态历史 Cups **100%** 但计数任务失败。架构上 GRU / Q-Former 替换 VAE 分别平均 **33.3% / 57.4%**，低于 VAE **83.3%**。token 数在 Wipe 上 **K=8** 最优；过大（16/32）破坏 π₀.₅ 动作专家对 token 预算的先验。
- **对 wiki 的映射：**
  - [FM-VLA](../../wiki/entities/paper-fm-vla.md) — 消融解读。
  - [KEMO](../../wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md) / [EventVLA](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) — **视觉稀疏记忆** vs **力觉压缩记忆** 分界。

### 5) 局限

- **链接：** arXiv §5
- **摘录要点：** 固定 **8-token** 瓶颈对「数百次接触」超长程可能不够；VAE 仅在本任务 demo 力数据上预训练，大规模异构 F/T 语料预训仍是开放方向。**代码截至入库日尚未发布**。
- **对 wiki 的映射：**
  - [FM-VLA](../../wiki/entities/paper-fm-vla.md) — 局限与开源状态。
