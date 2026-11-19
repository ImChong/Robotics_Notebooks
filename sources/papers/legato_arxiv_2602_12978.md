# Learning Native Continuation for Action Chunking Flow Policies（Legato）

- **类型：** 论文（RSS 2026）+ 官方项目页与仿真源码
- **arXiv：** <https://arxiv.org/abs/2602.12978>（HTML 全文：<https://arxiv.org/html/2602.12978v2>；PDF：<https://arxiv.org/pdf/2602.12978v2>）
- **版本与会议：** `[v1]` 2026-02-13；`[v2]` 2026-05-17（已录用至 **RSS 2026**）
- **作者：** Yufeng Liu$^{1,2}$、Hang Yu$^{2,4}$、Juntu Zhao$^{1,2}$、Bocheng Li$^{2,5}$、Di Zhang$^{2,4}$、Mingzhu Li$^{2}$、Wenxuan Wu$^{2}$、Yingdong Hu$^{3}$、Junyuan Xie$^{2}$、Junliang Guo$^{2\ddagger}$、Dequan Wang$^{1\dagger}$、Yang Gao$^{2,3\dagger}$（$\ddagger$ Project leader，$\dagger$ Corresponding authors；一作致谢称工作完成于 Spirit AI 实习期间）
- **机构：** $^1$ 上海交通大学、$^2$ 千寻智能（Spirit AI）、$^3$ 清华大学、$^4$ 同济大学、$^5$ 中国科学技术大学
- **项目主页：** <https://lyfeng001.github.io/Legato/>
- **官方源码：** <https://github.com/lyfeng001/Legato-kinetix>（MIT License；基于 `Physical-Intelligence/real-time-chunking-kinetix` 开发）
- **入库：** 2026-11-19
- **核对范围：** 通读 `arXiv:2602.12978v2` HTML 全文（摘要、§I 引言、§II 相关工作、§III 方法与推导 Eq.(1)–(16)/Algorithm 1、§IV 全部真机实验/消融/Kinetix/部署指南 Tables I–VII、§V 结论与附录 A–J Tables A.1–A.4）；读取 `lyfeng001/Legato-kinetix` 的 `LICENSE`、`README.md`、`src/model_legato.py`、`src/train_flow_legato.py` 与 `src/eval_flow_legato_versions.py`（未运行 Kinetix JAX 训练脚本，未取得论文中非公开的真机 `pi0.5`/`pi0`/`GR00T` 源码）。
- **通读：** 已完成
- **沉淀到 wiki：** [`wiki/entities/paper-legato.md`](../../wiki/entities/paper-legato.md)

---

## 核心摘要与问题定义

- **动机与痛点（§I、§III-B）：** Action chunking 的流匹配（Flow Matching）VLA 在真机异步执行时，由于推理延迟与生成多模态特性，相邻动作块边界极易出现跳变。推理期 [Real-Time Chunking (RTC)](../../wiki/entities/paper-real-time-chunking.md) 通过每步 $\Pi\mathrm{GDM}$ pseudoinverse guidance 约束旧块交叠部分，但它不改变策略训练分布，推理时需反向传播求 vector-Jacobian product（VJP）且容易在块间产生**伪多模态切换（spurious multimodal switching）**、犹豫停顿与局部高频纠偏。[Training-Time RTC (T-RTC)](../../wiki/entities/paper-training-time-real-time-chunking.md) 把硬掩码前缀放进训练，但仅锚定已提交前缀而不直接塑造持续软斜坡动力学。
- **关键实证前提（§III-B & Table A.2）：** 若只在去噪初始步 $t=0$ 做一次性的引导初始化（one-shot guidance）而后续步不引导，交叠区在多步积分中仍会逐渐漂移偏离已知动作（Table A.2 Pour 任务 Overlap RMSE 从 Legato 的 `5.14 ± 0.17` 飙升到 One-shot 的 `12.69 ± 1.55`），因此延续引导必须 **per-step（在每个去噪步之前反复施加）** 且同时满足 **训练–推理动力学一致性（training-inference consistency）**。

---

## 方法机制与公式（§III-C–D，与 Kinetix 代码口径对照）

1. **三段调度权重 $\omega$（§III-A2 / III-C）：**
   - 动作块长度 $H = r + s + d$，其中 $d$ 为完全受控的推理延迟前缀，$r$ 为线性衰减斜坡长度，$s$ 为每轮执行步长（注：$\omega \in [0,1]^H$：前 $d$ 步 $\omega_i=1$，中段 $r$ 步线性降至 $0$，剩余 $s$ 步为 $0$）。
2. **Action–Noise Mixture 与 Per-step Repeated Guiding（§III-C1–C2）：**
   - 引导初始分布定义等效噪声混合 $\epsilon_{\mathrm{eff}} = (1-\omega)\odot \epsilon + \omega \odot A$，插值路径 $Y_t = (1-t)\epsilon_{\mathrm{eff}} + tA = (1-\omega)\odot X_t + \omega \odot A$，与推理期每步施加的投影/引导操作符 $\mathcal{G}_{\omega}(Y) = (1-\omega)\odot Y + \omega \odot A$ 严格对齐。
   - **等效连续 ODE 与离散一步一致（Eq. 8–12）：** 当欧拉离散步长 $\Delta t = 1/N$ 时，若每步先执行 $Y_t \leftarrow \mathcal{G}_\omega(Y_t)$ 再前向预测 $f_\theta(Y_t, t)$，则会引发等效离散恢复力，其等效连续动力学为：
     $$
     \dot{Y}(t) = (1-\omega)\odot f_\theta(Y(t), t) - \kappa \odot (Y(t) - A), \qquad \kappa \triangleq \frac{\omega}{\Delta t}
     $$
     推导得出训练时的**重塑目标速度场**闭式表达（Eq. 14）：
     $$
     v_{\mathrm{target}}(\epsilon, A, t, \omega) = \bigl(1 - \kappa \odot (1 - t)\bigr)\odot (A - \epsilon)
     $$
     **核心收益：** 不需要改变模型推理接口形状，也不需要每步运行高代价 VJP 反向传播；模型直接学出能在每步重复引导后精准落回动作流形的速度场。
3. **随机化调度与部署启发式（§III-D、§IV-F Eq. 16）：**
   - 训练期随机采样 $d \sim \mathrm{Uni}[0, 10], r \sim \mathrm{Uni}[0, 50]$（真机主设定 $H=60, 30\,\mathrm{Hz}, N=5$）。
   - 论文原始架构把调度 $\omega$ 追加在条件向量行（`(H, D_a + 1)` 的条件张量），但 §IV-D3 与 **§IV-F 部署指南明确建议实际部署采用「`w/o cond. row`（去条件行）+ 无条件依赖训练策略」**，理由是当测试推理步数 $N_{\mathrm{infer}} \neq N_{\mathrm{train}}$（例如少步 $N_{\mathrm{infer}}=1$ 极限降延迟场景）时，`cond. row` 会导致调度权重跨步数失效；而 `w/o cond. row` 配合推理期公式 (16) 的有效权重缩放到固定学习动力学下可直接跨 $N_{\mathrm{infer}}$ 工作：
     $$
     \omega'_i = \mathrm{clamp}\!\left(1 - (1 - \omega_i)\frac{N_{\mathrm{train}}}{N_{\mathrm{infer}}},\; 0,\; 1\right)
     $$
4. **⚠️ 与开源 Kinetix 代码（`src/model_legato.py`）的实现差异核对：**
   - 论文正文的 $\omega$ 是**延续引导权重**（前缀为 1、后段为 0），而 `src/model_legato.py` 中的变量 `w` 是**噪声保留权重**（`warmup` 前为 `0.0`，之后为 `1.0`，对应 $1-\omega$）。
   - 代码中 `kappa = (1.0 - w) / dt`，`u_t = (action - noise) * (1.0 - kappa * (1 - time_expanded))`，`guided_x_t = (1.0 - w) * action + w * x_t`，与论文 Eq. (14) 数学等价；但针对 Kinetix 短 horizon（$H=8$），`build_weight_curve` 仅构造二值阶跃掩码（不含渐降斜坡 $r$），且 `__call__` 将 `w` 拼接在特征末维 `action_dim + 1 + obs_dim` 输入 `in_proj`。

---

## 实验数据与关键结论（arXiv:2602.12978v2 Tables I–VII, A.1–A.4）

- **真机硬件与协议（Appendix B / H / I）：** 双臂 7-DoF+夹爪桌面系统（总计 14-D 末端位姿+夹爪动作空间，头戴+双腕共 3 台 RGB 相机，RTX 4090 上 $\pi_{0.5}$ 单次前向约 170 ms；$H=60, f=30\,\mathrm{Hz}, N=5$；主实验显式控制延迟为 $d=8 \approx 266.7\,\mathrm{ms}$）。
- **主实验（Table I，vs RTC on $\pi_{0.5}$，均以论文报告为准）：**
  - **Bowls（50 trials）：** 任务分 `8.68 → 9.08`，时间 `52.88s → 42.66s`（↓19.3%），NSPARC `1.82 → 1.63`。
  - **Pour（难度最高三连倒水）：** 任务分 `9.34 → 9.72`，时间 `95.07s → 75.73s`（↓20.3%），NSPARC `2.85 → 1.65`，Chunk-overlap RMSE `7.64 → 5.14`。
  - **PickPlace：** 完成时间 `35.53s → 30.37s`；**Drawer：** `25.97s → 21.80s`；**Towel：** `25.93s → 20.00s`；5 项任务平均完成时间与轨迹平滑度均提升约 10%+，且不牺牲成功率/得分。
- **对比 T-RTC（§IV-C，Pour）：** Legato 得分 `9.72` vs T-RTC `9.46`，完成时间 `75.73s` vs `81.73s`，NSPARC `1.65` vs `2.46`。
- **对比 Temporal Ensembling（TE）与更难任务（§IV-D5、Table A.3）：** TE 在长延迟和多模态任务下产生明显拉扯迟滞；在 Skewer Fruit / Insert Pen 与动态反应测试（Table A.3，主动拉球，$d=s=8, r=22$）中，Legato 维持 `0.7` 成功率的同时将 NSPARC 从 RTC 的 `1.97 ± 0.14` 压低至 `1.48 ± 0.18`，未牺牲反应性。
- **跨模型迁移（Table V）：** 在 $\pi_0$ 与 **NVIDIA GR00T N1.6** 上同样优于基线，表明机制不局限于单一 VLA 底座。
- **Kinetix 仿真（§IV-E）：** 12 关卡多延迟评测中，当 $d > 0$ 时总体优于 `Inference-Time RTC`、`BID` 与 `Hard Masking`。

---

## 对 wiki 的映射

- 独立论文实体：[Legato｜流匹配 VLA 的训练期原生动作延续机制](../../wiki/entities/paper-legato.md)
- 公司实体与路线：[千寻智能（Spirit.AI）](../../wiki/entities/spirit-ai.md) · [Action Chunking](../../wiki/methods/action-chunking.md)
- 纵深路线：[VLA Stage 4 部署与系统整合](../../roadmap/depth-vla.md)
