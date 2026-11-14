# τ₀-VLA（分层机器人基础模型 + 世界模型引导测试时计算）

> 来源归档（ingest）

- **标题：** τ₀-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.16885>（v1，2026-08-17）
  - <https://tau0-vla.github.io/tau0-vla.pdf>
  - <https://tau0-vla.github.io/>
- **代码：** <https://github.com/sii-research/tau-0-vla>
- **权重：** <https://huggingface.co/sii-research/tau-0-vla>（低层），<https://huggingface.co/sii-research/tau-0-vla-libero>（LIBERO），<https://huggingface.co/sii-research/tau-0-vla-proposal>（高层 Proposal），<https://huggingface.co/sii-research/tau-0-vla-world-model>（高层 World Model LoRA）
- **作者与机构：** Xiaowei Cai、Jinyu Zhang（共一）、Yi Liu（共一）、Jianlan Luo（通讯）等；上海创智学院（Shanghai Innovation Institute）、智元机器人 Finch 团队（Agibot Finch）、香港中文大学（The Chinese University of Hong Kong）
- **发布日期：** 2026-07-27（项目页首发）/ 2026-08-17（arXiv v1）
- **入库日期：** 2026-08-19
- **核对日期：** 2026-11-14
- **通读：** 已完成
- **核对范围：** 全文通读 `2608.16885.txt`（Abstract、§I–§VII、Eq. 1–20、Algorithm 1、Fig. 1–5、Table I–VI、Appendix A–C.J）及 arXiv HTML（`2608.16885.html`，交叉核对 Table I 跨列对齐）；核对 Qwen3.5-9B / Step1X-Edit / Qwen3.5-2B + MoT 模型规格、40 维统一动作表、40.4M 高层记忆扰动样本配比、30 Hz/1 Hz 异步双系统调度与真机 Table I–III 数值。
- **一句话说明：** 分层 VLA 基础模型：以 Qwen3.5-9B 构成的可修订记忆 Proposal/Value/Reflective 模型配对 Step1X-Edit 世界模型在子任务边界执行置信路由 **测试时计算（TTC）beam search**；低层采用 **Qwen3.5-2B + MoT 动作专家**（`H=30` 步条件 flow matching、40 维统一接口），在 **40,115 小时**异构真机数据与多模态数据上 3 阶段训练，将 4 项 13–25 步长程真机平均成功率从直出的 `27.50%` 抬升至分层 Plan Once `45.00%`，并支持跨本体执行。

## 核心摘录（全文核对）

### 1) 高层策略与世界模型引导测试时计算（§IV-B、Algorithm 1、Appendix C-J）

- **推理环与路由：** 在逻辑推理步 `t`，高层上下文 `h_t = (ℓ, M_{t-1}, z^*_{t-1}, o_t)` 输入 Proposal `P`（`Qwen3.5-9B`）产出直接子任务 `z_dir` 与观测对齐的执行记忆 `M_t`。根据同一前向的 token 概率与 logit 边缘路由：
  $$u_t^{\mathrm{all}} = \frac{1}{|\mathcal{I}_{\mathrm{all}}|}\sum_{i\in\mathcal{I}_{\mathrm{all}}} p_i,\qquad u_t^{\mathrm{mem}} = \frac{1}{|\mathcal{I}_{\mathrm{mem}}|}\sum_{i\in\mathcal{I}_{\mathrm{mem}}} \bigl(\lambda_i^{(1)} - \lambda_i^{(2)}\bigr),\qquad g_t = \mathbf{1}\bigl[u_t^{\mathrm{all}} \le \delta_{\mathrm{all}} \lor u_t^{\mathrm{mem}} \le \delta_{\mathrm{mem}}\bigr]$$
  当 `g_t=0` 时走快路径直接执行 `z_dir`；当 `g_t=1` 时进入 `Search(h_t, P, W, V, N, B, D)`。
- **Propose–Predict–Evaluate + Reflect：**
  - **Proposal (`P`)、Value (`V`)、Reflective (`F`)：** 均由同一机器人预训练 `Qwen3.5-9B` VLM checkpoint 独立微调。搜索树从 `b_root` 起逐层扩展至深度 `D`，每条保留分支采样 `N` 个候选文本子任务。
  - **World Model (`W`)：** 基于 `Step1X-Edit` 训练，仅以单视角 **head-camera RGB** 与候选子任务 `z` 预测完成该子任务后的终端头部图像 `ô = W(õ, z)`。
  - **Value (`V`)：** 表单化为五档序数 VQA（从明显错误到明显正确，映射到 `[0.05, 0.95]`），对 `(ℓ, z, ô)` 打分并全局保留累积得分前 `B` 条分支（branch-local 记忆不污染持久化 `M_t`）。
  - **Reflective (`F`)：** 接收观测对齐真实上下文 `h̄_t = (ℓ, M_t, z^*_{t-1}, o_t)` 与深度 `D` 保留束 `C_t`，自回归产出最终子任务 `z^*_t`（可修复或重构束外动作）。

### 2) 可修订执行记忆与反污染质控（§V-C、Appendix C-E、Table VI）

- **自动合成 5 类记忆扰动家族：** 利用 Gemma4-31B-it 从三视角关键帧（`top_head`、`hand_left`、`hand_right`）生成 `<think>/<memory>/<subtask>` 标注，并在 `88.26%` 清洗率后保留 **40.4M** 条样本（剔除 `11.74%` 脏数据；两规则名词表跨任务防污染过滤器把 3.26M 验证集污染压至零，丢弃率仅 `0.42%`）：
  - `within-subtask`（58%，`M_n → M_n`）；`transition`（15%，`M_n → M_{n+1}`）；`catch-up`（10%，`M_{n-1} → M_n` 修复滞后）；`rollback`（12%，`M_{n+1...n+3} → M_n` 限幅 ≤10–15% 防止过度怀疑正确记忆）；`error-think`（5%，失败帧触发重试或回滚）。

### 3) 通用低层 VLA、40 维统一接口与三阶段训练（§IV-C、§V-A/B、Appendix C-B/D/F）

- **低层结构：** `Qwen3.5-2B` VLM 骨干 + `MoT`（Mixture-of-Transformers）动作专家，通过 40 维掩码条件流匹配（`H=30`、10 步均匀 Euler 积分、`τ = 0.001 + 0.999x, x ~ Beta(1.5,1)`，单样本归一化损失上限 100）输出相对当前状态的 EEF 位姿（位置含夹爪坐标系 delta + `Rot6D`）或相对关节偏移。
- **40 维布局（Table IV）：** `1–3` 左 EEF pos、`4–9` 左 Rot6D、`10–12` 右 EEF pos、`13–18` 右 Rot6D、`19/20` 左右夹爪、`21–22` 腰部、`23–24` 平面底盘速度、`25–32` 左臂关节、`33–40` 右臂关节；通过状态/动作对角 mask `M` 与文本控制元提示 `η`（`Robot type / Control mode <eef/eef_wbc/joint> / Whole-body control`）跨平台共享参数。
- **40,115 h 数据配比（Appendix C-D）：** 内部真机 **23.4K h**（AGIBOT G1 `21.9K h`、AGIBOT G2 `585 h`、ARX AC One `578 h`、Franka `347 h`）+ 公开数据 **16.7K h**（含 `9.25K h` 开源 UMI 数据）+ 多模态 VLM 共训数据。
- **三阶段训练与异步推理（§V-B、Appendix C-F）：** Stage 1 知识隔离（KI）预训动作专家 → Stage 2 解冻 VLM 端到端多模态共训 → Stage 3 任务微调。部署采用异步双系统：后台高层每 **~1 s** 刷新单 episode 缓存子任务，低层按 **~30 Hz** 无阻塞读取执行。

### 4) 真机评测与 TTC 结果（§VI、Table I–III、Fig. 4–5、Appendix C-C/I）

- **长程真机 Table I（AGIBOT G1，各 10 trials，最限 20/10 min）：**
  - Clean Room（25 步）：GR00T N1.7 `0/10 (59.80%)`、LingBot-VLA `0/10 (66.60%)`、π₀.₅ `4/10 (86.20%)`、τ₀-VLA 直出 `4/10 (92.80%)`、**τ₀-VLA 分层 Plan Once `5/10 (94.80%)`**；
  - Prepare Ingredients（14 步）：`1/10 (68.57%)` / `0/10 (35.00%)` / `2/10 (73.93%)` / `2/10 (66.43%)` / **`4/10 (82.86%)`**；
  - Tomato and Egg Stir Fry（22 步）：`0/10 (24.32%)` / `0/10 (12.27%)` / `0/10 (49.77%)` / `0/10 (65.00%)` / **`4/10 (81.82%)`**（重复加盐即判失败并扣 0.5）；
  - Make Milk Tea（13 步）：`0/10 (28.46%)` / `0/10 (63.85%)` / `3/10 (82.31%)` / `5/10 (96.15%)` / **`5/10 (91.92%)`**；
  - **四任务均值：** GR00T N1.7 `2.50% (45.29%)`、LingBot-VLA `0.00% (44.43%)`、π₀.₅ `22.50% (73.05%)`、τ₀-VLA 直出 `27.50% (80.10%)`、**分层 Plan Once `45.00% (87.85%)`**。
- **跨本体直出 Table II（10 trials，ARX AC One Collect Laundry & bimanual Franka Research 3 Tidy Makeup Table）：**
  - τ₀-VLA 达成 **Collect Laundry `10/10 (97.00%)`**（π₀.₅ `9/10`）、**Cotton Pad `10/10 (95.00%)`**、**Eyelash Curler `9/10 (92.50%)`**、**Makeup Puff `10/10 (95.00%)`**。
- **TTC 开环与闭环（Fig. 4–5、Table III、GPT-5.4 majority-vote judge）：**
  - OOD Book Organization 下一子任务准确率：**TTC `74.0%`** vs Best-of-N `57.5%` vs Plan Once `50.0%`；Make Milk Tea Plan Once `64.7%`（Fig. 5a）、Book Organization 综合 Plan Once `55.3%`（Fig. 5b），随算力（PFLOPs/sample）快速爬升后渐饱和。
  - **真机闭环 Table III（Plan Once → TTC）：** Make Milk Tea `5/10 (91.92%) → 7/10 (95.38%)`、Book Organization `6/10 (66.67%) → 9/10 (93.33%)`、Clean Room `5/10 (94.80%) → 7/10 (97.60%)`。

## 对 wiki 的映射

- 同步升级 [`wiki/entities/paper-tau0-vla.md`](../../wiki/entities/paper-tau0-vla.md)，纠正旧版 Table III `Clean Room 6→9/10`（实为 Book Org `6→9/10`、Clean Room `5→7/10`）与底层模型版本（`Qwen3.5-2B` + `Qwen3.5-9B` + `Step1X-Edit`），并与 [`sii-research/tau-0-vla`](../repos/sii_research_tau_0_vla.md) 的 2026-09-21 开源进度严格对齐。
