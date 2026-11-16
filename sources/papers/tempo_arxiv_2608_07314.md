# arXiv:2608.07314 — TEMPO: Semantic-Action Decoupled RL Post-Training for Vision-Language-Action Models

> 原始论文归档（v2 完整核对版）

- **标题：** TEMPO: Semantic-Action Decoupled RL Post-Training for Vision-Language-Action Models
- **作者：** Ziheng Liu（浙江工商大学 Zhejiang Gongshang University）, Quantao Yang（瑞典皇家理工学院 KTH Royal Institute of Technology；Qualia Robotics, Denmark）
- **链接：** <https://arxiv.org/abs/2608.07314>（本文核对版为 arXiv v2，2026-11-12；`2608.07314.txt`）
- **项目页：** <https://anonymous.4open.science/w/tempo-page/>（2026-11-14 原件抓取核实返回 HTTP 404，匿名托管站点已失效/未开放可访问实现）
- **开源状态：** 未开源（仅匿名项目页链接，抓取时返回 HTTP 404，未提供可运行源码或权重；注：勿与另一同名工作 `tempo-robot/TEMPO` [`arXiv:2609.16864`] 混淆）
- **通读：** 已完成
- **核对范围：** 已通读 `2608.07314.txt` 全文（Abstract、Sec. I–V、Eq. 1–15、Fig. 1–4、Table I–VII 与 References [1]–[35]）。

---

## 核心动机与算法数学细节（Sec. I, III, Eq. 1–14）

针对 VLA 策略（以 **FLOWER** 为主实验底座，并在 v2 扩展验证 **VLA-Adapter-Pro**），现有在线 RL 后训练将模块全部塞入单一 RL 循环、以相同步频联合更新。然而多模态语义映射与底层连续动作控制具有截然不同的适配特性：若语义映射层 $\pi_{\theta_s}$ 频繁更新，供给动作专家的条件隐动作 $z_t$ 将剧烈漂移（latent-action drift），迫使动作专家追逐移动目标；若完全冻结语义适应又会限制视觉-语言对齐改善空间。

1. **三段解耦架构（Eq. 2）：**
   $$h_t = \text{VL}(o_t, \ell_t),\qquad z_t = \pi_{\theta_s}(h_t),\qquad a_t = \pi_{\phi_a}(z_t)$$
   其中预训练视觉-语言骨干 $\text{VL}$ **完全冻结**，仅分别优化语义映射层 $\pi_{\theta_s}$ 与底层动作专家 $\pi_{\phi_a}$。
2. **双模块独立 TD3 闭环（Eq. 3–12）：**
   - **语义级 TD3（Semantic-Level TD3）：** 回放缓冲区 $D_S = \{(h_t, z_t, r_t, h_{t+1}, d_t)\}$；孪生 Critic $Q^s_1, Q^s_2(h_t, z_t)$；目标动作 $\tilde{z}_{t+1} = \bar{\pi}_{\theta_s}(h_{t+1})$；Bellman 目标 $y^S_t = r_t + \gamma(1 - d_t)\min_{i=1,2}\bar{Q}^s_i(h_{t+1}, \tilde{z}_{t+1})$；Actor 损失 $L_s = -\mathbb{E}[Q^s_1(h_t, \pi_{\theta_s}(h_t))]$。梯度**仅更新 $\theta$**，不回传至冻结 VLM 或动作专家。
   - **动作级 TD3（Action-Level TD3）：** 回放缓冲区 $D_A = \{(h_t, z_t, a_t, r_t, h_{t+1}, d_t)\}$；孪生 Critic $Q^a_1, Q^a_2(z_t, a_t)$；目标链路 $\tilde{z}_{t+1}=\bar{\pi}_{\theta_s}(h_{t+1}),\ \tilde{a}_{t+1}=\bar{\pi}_{\phi_a}(\tilde{z}_{t+1})$；更新动作专家时以截断梯度的当前隐动作 $z_t^{\text{sg}} = \operatorname{sg}(\pi_{\theta_s}(h_t))$ 输入 $L_a = -\mathbb{E}[Q^a_1(z_t^{\text{sg}}, \pi_{\phi_a}(z_t^{\text{sg}}))]$，梯度**仅更新 $\phi$**。
   - **稀疏终止边界奖励：** $r_t = 1$ 仅在当前指令完成时触发，且在长程多指令链的每个子指令边界执行 $d_t = 1$ 截断 Bellman 自举。
3. **双时间尺度频率解耦（Eq. 13–14）：**
   设每轮动作与语义 Critic 更新次数分别为 $N_a, N_s$，延迟 Actor 更新间隔分别为 $d_a, d_s$，则有效更新频率比为：
   $$\rho = \frac{f_a}{f_s} = \frac{N_a / d_a}{N_s / d_s} = \frac{N_a d_s}{N_s d_a}$$
   当 $\rho > 1$（实测推荐 $4:1$ 至 $5:1$）时，动作专家高频吸收交互控制反馈，语义映射低频稳态更新，显着抑制 $z_t$ 漂移。

---

## 实验配置与完整定量结果（v2 Tables I–VII & Fig. 3–4）

### 1. CALVIN ABC→D 主对比（Table I，3 种子）
- **FLOWER（预训练参照）：** SR1–SR5 `99.4 / 95.8 / 90.7 / 84.9 / 77.8`，Avg. Len. `4.49`
- **FLOWER-RL（单循环同频 TD3）：** `99.43±0.35 / 97.03±0.58 / 91.87±0.74 / 85.37±0.61 / 78.70±0.70`，Avg. Len. `4.52±0.028`
- **DeFI（表内最强外部基线）：** `97.9 / 94.2 / 90.7 / 87.0 / 81.2`，Avg. Len. `4.51`
- **TEMPO（双环双频 TD3）：** **`99.83±0.15 / 96.77±0.35 / 92.77±0.15 / 87.30±0.20 / 81.70±0.40`**，Avg. Len. **`4.58±0.005`**（最佳单组运行见 Table III `5:1`：`100.0 / 97.1 / 92.9 / 87.1 / 81.7`，Avg. Len. `4.59`）。

### 2. 组件消融与更新频率比（Tables II–IV，1M 环境步）
- **Table II（组件消融）：** `Ours (full)` SR5 `81.7` / `4.59` > `w/o expert update` (`79.6 / 4.55`) & `w/o projection update` (`79.8 / 4.54`)。
- **Table III–IV（$\rho = f_a:f_s$，均 6,600 critic / 3,300 actor 更新，1M env steps）：**
  - `1:1` (`3,300/3,300` critic, `7.387` PFLOPs)：SR5 `77.7` / Avg. Len. `4.46`（反而不如 FLOWER 4.49）；
  - `5:1` (`5,500/1,100` critic, `12.276` PFLOPs)：SR5 **`81.7`** / Avg. Len. **`4.59`**；
  - `10:1` (`6,000/600` critic, `13.387` PFLOPs)：SR5 `81.2` / Avg. Len. `4.57`。

### 3. 隐动作漂移 $D_{\text{drift}}$ 与目标任务成功率（Table V，`lift_blue_block_slider`，50 trials × 360 steps）
$$D_{\text{drift}} = \frac{1}{N}\sum_{i=1}^N \|\hat{z}_i^{\text{trained}} - \hat{z}_i^{\text{FLOWER}}\|_2 \quad (N = 108)$$
- `FLOWER`: `0.000` (`66.0%`) | `FLOWER-RL`: `0.611±0.041` (`68.0%`) | `1:1`: `0.641±0.034` (`66.0%`)
- `2:1`: `0.563` (`74.0%`) | `3:1`: `0.470` (`72.0%`) | **`4:1`: `0.409±0.025` (`98.0%`)** | **`5:1`: `0.424±0.027` (`96.0%`)**。

### 4. 任务集规模、架构通用性与真机三任务（Tables VI–VII, Fig. 4）
- **Table VI（10:1 下 CALVIN 训练子任务数 1/2/4/7/34）：** 单任务 SR5 `81.2 / 4.57`，扩展至 34 项全集反而回落至 `79.2 / 4.54`（但仍高于 FLOWER），表明盲目扩充后训练任务集并不单调有益。
- **Table VII（架构泛化 — `VLA-Adapter-Pro`，5:1，3 种子）：** 原模型 SR5 `77.23±1.31%` (`4.437±0.023`) → 接入 TEMPO 后 **`79.17±0.78%` (`4.487±0.009`)**。
- **真机实验（Sec. IV-C, Fig. 4）：** 6-DoF **Synria Alicia-D** 机械臂 + 单自由度夹爪 + Intel RealSense D435i（第三人称）& D405（腕部）；每任务采集 60 条演示（总计 180 条）用于预热，开展 **3 项多阶段任务**（block sliding、car retrieval、带黄色积木遮挡物的 car placement），每个检查点每种子评测 20 次取累积二值奖励，TEMPO 后期回报显著且稳定高于 FLOWER-RL（尤其在抽屉未完全打开或方块遮挡等需要多步前置清理的场景）。
