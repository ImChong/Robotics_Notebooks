# vla.simd（arXiv:2609.24274）

> 来源归档（paper）

- **标题：** vla.simd: Efficient CPU Inference for Language-Conditioned Manipulation
- **类型：** paper
- **作者：** Khanh D. Nguyen, Hoang M. Truong, An T. Le
- **机构：** VinRobotics (Vietnam), Center for AI Research (VinUniversity), Intelligent Autonomous Systems (TU Darmstadt)
- **arXiv：** <https://arxiv.org/abs/2609.24274>（核对版本 `arXiv:2609.24274v1`，2026-09-21 提交）
- **PDF：** <https://arxiv.org/pdf/2609.24274>
- **项目页：** <https://vla-simd.github.io/>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 提出面向 CPU 端操作策略推理的 `vla.simd` 引擎（共享 SIMD 微内核、按生命周期计算复用与离线寄存器 tile 选择）、动作供给预算判据 `C1/C2` 及轻量语言条件策略 `IMPACT`，首次在被动散热树莓派 Pi 5 上实现 `33.5`（fp32）/ `81.2`（int8）actions/s 的持续动作供给。
- **通读：** 已完成
- **核对范围：** 通读 arXiv `2609.24274v1` 全文（Abstract, §§I–VIII, Algorithm 1, Tables I–VII, Figures 1–5, References [1]–[47]），并对照 `2609.24274.html` 核验 Table III（动作供给率与 Pi 5 90 秒热浸后数据）与 Table V（LIBERO 成功率）及 `.abs.html` 版本元数据。

## 开源状态

- **双盲匿名审稿链接可见、正式 GitHub 尚未发布**（2026-11-14 原文与项目页联合核查）：论文正文链接到项目页 `https://vla-simd.github.io/`，项目页未列正式 GitHub 仓库，但提供匿名双盲代码仓库 `https://anonymous.4open.science/r/vla-simd` 与 `https://anonymous.4open.science/r/lerobot-vla-simd`，并在页脚说明 *"Code and checkpoints will be released on acceptance."*；目前尚未发布实名 GitHub 源码与权重，本知识库本地 `originals/repos/` 亦未归档该匿名仓库实现。

## 全文核心核查摘录

- **动作供给预算分析（§III, Eq. 1）：**
  - 区分 **动作供给率 $f_{\mathrm{eff}} = H / T_q$** 与真实闭环**反馈频率**（每次执行动作基于至少 $d = \lceil f_c T_q \rceil$ 步前的观测）。
  - **滞后执行（Lagged, `C1`）：** 到达后从新 chunk 第 1 步开始执行，要求 $d \le H$（即 $f_{\mathrm{eff}} \ge f_c$，本文主设 $f_c = 30\,\text{Hz}$）。
  - **时间对齐执行（Time-Aligned, `C2`）：** 保留控制步对齐并丢弃过期前缀（如 RTC / LeRobot 异步客户端），要求 $2d \le H$（近似 $f_{\mathrm{eff}} \ge 2 f_c = 60\,\text{actions/s}$）。
- **`vla.simd` CPU 推理引擎（§IV, Algorithm 1, Fig. 2）：**
  - **零运行时深度学习框架开销：** 自包含 C/SIMD 推理管线处理图像缩放、归一化与文本分词。
  - **按生命周期分层复用：** 权重每进程（per-process）离线重排打包为连续 `fp32` 面板；指令相关编码与 FiLM 系数每回合（per-episode）缓存一次；每次查询仅算观测相关特征。
  - **共享打包面板 SIMD 微内核（SMK）：** 线性层、卷积（按单面板动态 `im2col` 构造图像块，避免完整 patch 矩阵爆内存）与注意力 QKV/输出投影统一复用同一 SMK；Apple Silicon（M4）默认调用 Accelerate，Intel/AMD AVX2 与 ARMv8 NEON 使用 SMK（注意：不整除 $n_r$ 的尾层如动作头走按行分块核）。
  - **离线寄存器 tile 选择（Algorithm 1）：** 基于寄存器上限 $R_{\max}$、FMA 与访存吞吐（$P_{\mathrm{fma}}, P_{\mathrm{ld}}$）及延迟 $\tau$，并用编译器 `SPILLS` 检查消除栈溢出：在给定层宽度最大公约数 $g=64$ 下，**AVX2** 选用 `6×16`（占 `15/16` 寄存器），**NEON** 选用 `4×16`（占 `24/32` 寄存器），在 Raspberry Pi 5 四线程达 `106.9 GFLOP/s`（L2 缓存单核理论算力 `96%`）。
  - **可选 `W8A8` 量化路径：** 对称 `int8` 权重（per-output-channel）+ 动态激活（线性层 per-token、卷积 per-tensor），32 位整数累加后转回 `fp32`（ARM `dotprod` / x86 `AVX-VNNI`）；规范化层、非线性层、不支持的输出宽度及缓存文本编码器保留 `fp32`。
- **`IMPACT` 语言条件策略（§V, Fig. 3）：**
  - 在 ACT 基础上接入冻结 `T5-small`（`35.3 M`，每次仅编码每条指令 `1.2 GFLOP/episode`），通过两路机制调控策略：**（1）** 投影最多 32 个文本 token 拼入 6 层 Transformer Encoder（`d=512`, 8 头, FFN `3200`）；**（2）** 对文本隐状态做均值池化并映射出各 ResNet-18 stage 的 FiLM 缩放/平移系数 $[\gamma_i; \beta_i] = G_i \bar{e} + b_i$。
  - 主策略体量 `60 M` 参数，输出 `50` 步动作块；在 SO-101（`480×640` 双视角，600 visual tokens）上每次查询约 `88 GFLOP`，其中语言条件仅占约 **`2.6%`**（文本 token 约 `2.3 GFLOP` + FiLM `9 MFLOP`），推理仅慢于无语言浅版 ACT `1.20–1.33×`。
- **实验结果核验（§VI, Tables I–VII）：**
  - **跨 6 策略 × 4 CPU（M4 / i9-14900HX / Ryzen 5 5500 / Pi 5）：** 相对 `torch.compile`（PyTorch 2.11/2.13）中位加速 **`~1.4×`**（范围 `0.83–3.37×`），数值相对参考 PyTorch 误差上限 **`8.8 × 10^-5`**（机器人物理单位），Intel 与 AMD 后端在测试输入上达成逐位一致（bitwise-identical）的 `fp32` 输出。
  - **树莓派 Pi 5 90 秒被动散热热浸测试（Table III）：** `IMPACT` 是唯一达标 `30 Hz` 门槛的语言条件策略——初始 `41 actions/s`（`Tq = 1.21 s`），90 秒持续推理热降频后仍维持 **`33.5 actions/s` (`fp32`, 满足 `C1`)** 与 **`81.2 actions/s` (`int8`, 满足更严苛时间对齐 `C2`)**；`SmolVLA`、`TurboVLA` 与 `Octo-Small` 在 Pi 5 均未过 `C1`（`6.1 / 9.3 / 7.8 actions/s`），但在桌面 `Ryzen 5 5500` 上 `vla.simd` 成功把 `Octo-Small` 和 `SmolVLA` 推过 `30 Hz` 线。
  - **CPU 参数消融分化（Table IV）：** 分块卷积对非苹果平台提速显著（i9 `+156%`、Ryzen `+260%`），而每视角额外线程只帮 M4（`-6.2%` 反向表示多视角线程更快即 B 更优）却拖慢 x86/Pi 5；`W8A8` 在 Pi 5 提速 `2.39×`（IMPACT）/`2.68×`（ACT），但在 M4 上使 `Octo-Small` 与 `SmolVLA` 反而减速至 `0.66×` 与 `0.71×`。
  - **LIBERO 仿真与指令打乱（Table V, GPU 训练评估，200 ep/suite 无机器人预训练）：** `IMPACT` 末轮 30k 检查点四套件均值 **`76.4%`**（Spatial `83.5` / Object `83.5` / Goal `84.5` / Long `54.0`；评估选优检查点 `78.6%`）；LIBERO-Goal 600 轮指令置换实验按新指令完成率 **`91%`**、误执行原指令 **`0%`**（对照正确指令 `88.5%`）。
  - **双真机部署（Tables VI–VII, 每行 20 trials）：** SO-101 多任务 `IMPACT` 在 `fp32` 下分别取得 `90% / 85% / 60%`，`int8` 取得 `85% / 95% / 50%`，长程抽屉任务 `65%`；UR10e + Robotiq 夹爪由 `vla.simd` 伺服 `SmolVLA`，M4（端到端往返 `1177 ms`）与 Ryzen 5 5500（`2343 ms`）均达到 `60% (12/20)`，且真实序列化与网络传输占去约 `0.5–1.0 s`。

## 对 wiki 的映射

- 实体主页：[`paper-vla-simd`](../../wiki/entities/paper-vla-simd.md)
- 上游关联主题：[`vla`](../../wiki/methods/vla.md)、[`action-chunking`](../../wiki/methods/action-chunking.md)、[`manipulation`](../../wiki/tasks/manipulation.md)、[`embodied-station-2026-10-18-26-papers-technology-map`](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
