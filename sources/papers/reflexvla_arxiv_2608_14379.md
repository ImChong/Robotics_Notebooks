# Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation（arXiv:2608.14379）

> 来源归档（ingest）

- **标题：** Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation
- **缩写 / 框架：** **ReflexVLA** / **ReflexBench**
- **类型：** paper / vla / latency / dynamic-manipulation
- **arXiv：** <https://arxiv.org/abs/2608.14379>（核对版本 `arXiv:2608.14379v1`，2026-08-14 提交；PDF：<https://arxiv.org/pdf/2608.14379>）
- **项目页：** <https://reflexvla.github.io/> — 归档见 [`sources/sites/reflexvla-github-io.md`](../sites/reflexvla-github-io.md)
- **基准仓库：** <https://github.com/LxRoboticsLab/ReflexBench>（BSD 3-Clause）— 归档见 [`sources/repos/reflexbench.md`](../repos/reflexbench.md)
- **数据集：** <https://huggingface.co/datasets/cyx337/ReflexBench_dataset>（LeRobot v3.0，约 200 ep/task）
- **通讯：** `chen_yuxuan@sjtu.edu.cn`
- **作者：** Yuxuan Chen、Wanruo Zhang、Xiao Li
- **机构：** 上海交通大学（SJTU）
- **入库日期：** 2026-08-17
- **最后更新：** 2026-11-14
- **一句话说明：** 为反应关键操纵构建解耦仿真的延迟感知基准 ReflexBench，并基于 VLA-Adapter（1B）提出“冻结 DINOv3 未来隐特征预测 + 视觉中间层因果 MHA 时序融合 + 批量视觉编码与 CUDA Graph 回放”框架 ReflexVLA。
- **同名分流：** 不是 ICML 2026 *Reflex: Real-Time VLA Control through Streaming Inference*（[arXiv:2607.14695](https://arxiv.org/abs/2607.14695)）。
- **通读：** 已完成
- **核对范围：** 通读 `arXiv:2608.14379v1` 全文（Abstract, §§I–VI, Eq. 1–11, Figures 1–6, Tables I–IV, References [1]–[22]），并与项目主页 `sites/reflexvla.html`、开源基准仓库 `LxRoboticsLab/ReflexBench` 及 `.abs.html` 版本记录交叉核验所有任务成绩、消融耗时、真机样本与开源范围差异。

## 开源状态（2026-11-14 交叉核查更新）

- **基准与数据集已开源；ReflexVLA 策略本体仍待录用后开放：**
  - **已开源部分：** 官网已放出 **ReflexBench** 基准与评测代码（<https://github.com/LxRoboticsLab/ReflexBench>，BSD 3-Clause，含 Isaac Lab 6 任务环境、cuRobo/RL 采数脚本、LeRobot 转换器与同步/异步延迟评测客户端）及 **ReflexBench_dataset**（<https://huggingface.co/datasets/cyx337/ReflexBench_dataset>，约 200 episodes/task）。
  - **未开源部分：** 官网的策略代码按钮 `Code` 仍为占位元素（`is-placeholder code-button`，`title="Code will be released after the paper is accepted"`），且经源码级检索确认 `LxRoboticsLab/ReflexBench` 不包含 ReflexVLA 的 1B 模型实现、DINOv3 未来监督、中间层时序融合或 CUDA Graph 推理服务端。

## 摘录 1：问题定义与 ReflexBench 基准（§I–§III）

- **痛点：** 静态操作基准忽略物体自身动态演化，且多数仿真在模型推理期间暂停物理引擎，完全抹除了“观测→推理→执行”时间差导致的位姿偏移。
- **ReflexBench 六个动态任务：** Conveyor Belt Pick-and-Place（传送带拣放，含干扰物）、Ball Catching（空中接球）、Whack-a-Mole（限时打地鼠）、Rolling Ball Interception（斜坡滚球拦截）、Ball Throwing（投球入筐）、Rotating Peg Insertion（连续旋转轴孔装配）。
- **混合采数管线（§III-B）：** 五个规划可达任务采用分阶段规划器结合位置/速度前瞻轨迹预测生成示教；传送带等复杂交互任务训练专用 RL 策略作为专家采数（实测仓库中即采用 skrl PPO 检查点）。
- **延迟感知评测协议（§III-C, Eq. 1, Fig. 2）：** 仿真步进与机器人控制解耦；**同步模式**在等待推理结果期间冻结机械臂并让环境继续演进，**异步模式**在执行上一周期动作块的同时推进下一轮推理；支持人工设定固定周期，也支持先在真机测得壁钟推理耗时，再通过实测实时倍率 $\mathrm{RTF} = t_{\mathrm{sim}}/t_{\mathrm{wall}}$ 将等价延迟毫秒精确注入仿真。

## 摘录 2：ReflexVLA 架构与加速（§IV, Eq. 2–11, Fig. 3）

- **骨干网络（约 1B）：** 基于 VLA-Adapter，采用 `224×224` 融合 `DINOv2 + SigLIP` 视觉编码器、`Qwen2.5-0.5B` 语言骨干与连续回归动作头，通过拼接在 BOS 之后的视觉 token 与可学习 action queries 输出动作块。
- **未来隐特征预测（§IV-A）：** 序列末尾追加与动作块等长（$H$ 个）的 learnable future tokens，投影后用掩码余弦损失（$\lambda_{\mathrm{future}}=0.05$）对齐**冻结 DINOv3** 编码的未来观测特征 $y_{t+i} = \phi_{\mathrm{DINOv3}}(o_{t+i}) \in \mathbb{R}^{1024}$。若开放未来目标参数联合微调，表示空间坍缩导致传送带 SR 由 `36.8%` 暴跌至 `4.9%`（冻结目标可升至 `62.8%`）。
- **视觉骨干中间层因果时序融合（§IV-B）：** 对 $V$ 视角、$T$ 历史帧提取中间层 patch 特征 $X_{\mathrm{mid}} \in \mathbb{R}^{B \times V \times T \times P \times D}$，在同一空间 patch $(v,p)$ 跨时间步先降维再做因果多头自注意力（`MHA_causal`），随后升维残差加到当前帧末层特征 $x^{\mathrm{final}}_{v,p,t}$。由于语言模型只接收融合后的当前帧 $V \times P$ 个 token，**不膨胀 LLM 上下文长度**。
- **低延迟系统优化（§IV-C）：** 将 $V \times T$ 张图像合并成单次批量 ViT 前向调用，避免多次独立调起；再将“视觉编码→时序融合→投影→LLM→动作头”整条静态计算图捕获为单张 `CUDA Graph`（式 11），推理期仅拷入预分配显存缓冲区并直接重放。

## 摘录 3：实验与消融（§V, Figs. 4–6, Tables I–IV）

- **基础协议：** 6 任务各 200 条演示共训单策略；ReflexVLA 用 `T=2` 连续历史帧、$\lambda_{\mathrm{future}}=0.05$；主对比采用异步推理、`chunk=8`、`horizon=2`，在单张 RTX 5880 Ada 上评测（每任务 150 episodes × 3 seeds）。
- **Q1 推理协议扫描（Figs. 4–5，SmolVLA）：** 低频时异步反而因观测-动作错位劣于同步；到 30 Hz 高频后异步反超（如 Rolling Ball Interception `30 Hz` 下异步 `84%` vs 同步 `46.7%`）；扫描显示 **大 chunk（如 8/16）+ 短执行地平线（horizon=1/2）** 配合高频异步表现最优。
- **Q2 ReflexBench 主表（Table I, %）：**
  - **ReflexVLA (1B):** 传送带 `73.8±4.1`、接球 `7.3±0.7`、打地鼠 `100.0±0.0`、斜坡截球 `77.1±7.1`、投球 `31.7±1.0`、旋转插销 `12.4±1.5`，均值 **`50.4%`**。
  - **对比基线：** VLA-Adapter (1B) `30.3%`、OpenVLA-OFT (7B) `36.0%`、$\pi_{0.5}$ (4B) `36.9%`、DynamicVLA (0.5B) `37.3%`、SmolVLA (0.5B) `38.3%`、PUMA (4B) `50.2%`。ReflexVLA 以 1/4 参数量追平 4B 动态专精模型 PUMA，相对骨干提升 `+20.1 pp`。
- **Q3 静态榜保留能力（LIBERO Table II）：** Spatial `98.2` / Object `99.2` / Goal `98.0` / Long `93.6` → **均值 `97.2%`**，与骨干 VLA-Adapter（`97.3%`）及 OpenVLA-OFT（`97.1%`）、$\pi_{0.5}$（`96.9%`）持平。
- **Q4 传送带消融（Table III）：** 基线 `36.8% / 81.522 ms` → 可训练未来目标 `4.9% / 82.508 ms` → **冻结未来目标 `62.8% / 82.508 ms`** → 末层交叉注意力 `66.1% / 127.335 ms` / 末层 MHA `68.2% / 125.824 ms` / **中间层 MHA `71.7% / 125.107 ms`** → **Batch ViT + CUDA Graph `73.8% / 64.991 ms`**（端到端耗时砍半）。
- **Q5 真机 AgileX Piper（Table IV，各 20 trials）：** Conveyor `16/20`（SmolVLA `2/20`、PUMA `13/20`）、PressButtons 30s 内均值 `22.5` 次（SmolVLA `0.9`、PUMA `20.8`）、CatchBalls 每 10 球接住 `6.7` 个（SmolVLA `3.8`、PUMA `5.4`）。
- **作者自述局限（§V-E）：** 未来预测与时序融合目前仅用于下游微调而非大规模机器人预训练；目前仅测了标准同步与朴素异步推理，未与 RTC 等更高阶流式异步解码结合。

## 对 wiki 的映射

- 实体主页：[`paper-reflexvla`](../../wiki/entities/paper-reflexvla.md)
- 回链：[`VLA`](../../wiki/methods/vla.md)、[`Action Chunking`](../../wiki/methods/action-chunking.md)、[`实时性↔泛化`](../../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md)、[`Manipulation`](../../wiki/tasks/manipulation.md)
