# Heracles: Bridging Precise Tracking and Generative Synthesis for General Humanoid Control（arXiv:2603.27756）

> 来源归档（ingest）

- **标题：** Heracles: Bridging Precise Tracking and Generative Synthesis for General Humanoid Control
- **类型：** paper / humanoid / motion tracking / diffusion / flow matching / recovery
- **arXiv abs：** <https://arxiv.org/abs/2603.27756>
- **arXiv v2：** <https://arxiv.org/abs/2603.27756v2>
- **PDF：** <https://arxiv.org/pdf/2603.27756>
- **项目页：** <https://heracles-humanoid-control.github.io/>
- **团队：** X-Humanoid Heracles Project Team（项目页列 Qiang Zhang 等；代码 Coming Soon）
- **入库日期：** 2026-05-25
- **通读：** 已完成
- **核对范围：** `2603.27756v2` 全文 §1–§6 及参考文献（含 Table 1–6、Fig. 1–7、Unitree G1 硬件部署与消融实验）
- **一句话说明：** **状态条件扩散中间件**插在高层参考运动与底层物理跟踪策略之间：状态贴近参考时近似恒等映射保零样本跟踪精度，大偏差时生成类人恢复关键帧轨迹，经闭环重规划喂给通用 tracker，无需显式 tracking/recovery 模式机。

## 摘要级要点

- **痛点：** 通用控制器多把运动控制写成**刚性参考跟踪**；名义条件下有效，但大扰动 / OOD 时盲目最小化即时 kinematic 误差 → 僵硬、非类人、难恢复跌倒。
- **对立面：** 纯生成式 BFM / URL 更自然但难保**时空精确跟踪**；需桥接「精确跟踪」与「生成式恢复」。
- **Heracles 定位：** **中间层**（middleware），非端到端单体策略，也非显式 FSM 切换；以机器人**实时本体状态**条件化扩散过程，**隐式**在 identity map 与 generative synthesizer 间过渡。
- **系统两层与控制频率（29-DoF Unitree G1，1.32 m / 35 kg）：**
  1. **生成中间件（25 Hz 重规划，$N_{\mathrm{exec}}=2$ 步 / $0.04\text{ s}$）：** 6 层 4 头 512 维 **AdaLN-Transformer（22.9M 参数）** 输入 $(\mathbf{p}_t, \mathbf{m}_t)$，通过条件 flow matching（$\Delta t=0.2\text{ s}$ 窗口内预测 $K=8$ 个关键帧残差 $\mathbf{r}_t$，$\bm{\tau}_t=\bm{\beta}_t+\mathbf{r}_t$，其中静态基线 $\bm{\beta}_{t,k}=\mathbf{p}_t$）生成短视界关键帧。
  2. **通用物理跟踪策略 $\pi$（50 Hz 推理 → 200 Hz PD/物理仿真）：** 接收 10 帧本体历史与经三次样条（关节）+ slerp（根朝向）密化的参考窗口（10 帧未来），用 **iFSQ（$f(x)=2.0\sigma(1.6x)-1$，$L=2K+1$ 奇数级含精确零状态）** 编码为离散运动 token $\mathbf{z}_d$，在非对称 PPO（单张 A100 80GB 上 16,384 并行环境）+ 核平滑自适应时间分箱采样下联合优化动作头与重建头。
- **关键训练与推理技巧：**
  - **首 token inpainting：** 固定 $x_t[0]$ 锚定当前状态，保障轨迹运动学连续。
  - **Directional Warm Start（SDEdit 风格）：** 从向目标线性插值的初值在 $t_{\mathrm{start}}=0.9$ 加噪后仅做 **5 步 Euler ODE 积分**即可推理。
  - **非对称噪声增强 & 雅可比加权：** 仅对起始本体状态 $\tilde{\mathbf{p}}_t=\mathbf{p}_t+\epsilon$ 加噪而保持参考 $\mathbf{m}_t$ 干净；用近似 $J^\top J$ 对角线权重按杠杆臂缩放损失，避免远端关节误差失配。
  - **双状态配置：** **38D**（29 关节 + 3D 根位置 + 6D 根朝向）与 **35D**（去全局根位置、仅靠本体/IMU 在线部署）。
- **核心量化结果（MuJoCo 101 条未见序列 + 真机 G1）：**
  - **通用跟踪（Table 4）：** 完成率 CR 达 **`90.6%`**，优于 VQ-VAE (`86.0%`)、MLP (`84.8%`)、SONIC (`79.3%`) 与无中间件独立 iFSQ (`87.2%`)；对比 iFSQ 根朝向误差降 `24.5%`（`0.3614 -> 0.2728 rad`）、高度误差降 `20.0%`（`0.0955 -> 0.0764 m`）。
  - **跌倒恢复子集（Table 5）：** CR 达 **`90.0%`**，大幅领先 VQ-VAE (`69.8%`)、iFSQ+H (`52.7%`)、iFSQ (`48.2%`)、MLP (`44.0%`) 与 SONIC (`42.8%`)，关节误差低至 `1.4114 rad`。
  - **关键消融（Table 6）：** 去掉非对称噪声增强 CR 跌至 **`78.6%`**（`-13.2%`，高度误差 `+91.5%`）；去掉雅可比权重 CR 跌至 **`82.1%`**（`-9.4%`）；去掉 directional warm start CR 跌至 **`87.2%`**（`-3.8%`）。
  - **真机 G1 验证（Fig. 4 & Fig. 7）：** 实现走/跑/踢腿/360° 踢/HOI 与仰卧/侧卧/俯卧**全向起身恢复**。

## 核心摘录（面向 wiki 编译）

### 与 SONIC / BeyondMimic / BFM 的对照（Related Work 归纳）

| 路线 | 代表 | 扰动下行为 | Heracles 差异 |
|------|------|------------|---------------|
| 规模化 tracking | SONIC, GMT, OmniXtreme | 刚性追参考，OOD 扭矩不可行 | 中间件按状态**改写**参考缓冲 |
| 跟踪 + 扩散引导 | BeyondMimic | 测试时 classifier guidance | Heracles 用**状态条件 flow**，闭环重规划 |
| 无参考 URL / BFM | BFM, BFM-Zero | 自然但难精确跟踪 | Heracles 保留 tracker 精度，生成仅补 OOD |
| 开环运动扩散 | MDM, HY-Motion | 无物理约束 | Heracles 输出给**物理 tracker** 执行 |

### 与 SD-AMP（arXiv:2605.18611）的分工

- **Heracles：** **分层**——生成层改参考轨迹，执行层仍是 tracking RL。
- **SD-AMP：** **单策略 RL**——训练期双判别器门控 AMP 风格，部署无中间件。

## 对 wiki 的映射

- 沉淀实体页：[Heracles（arXiv:2603.27756）](../../wiki/entities/paper-heracles-humanoid-diffusion.md)
- 项目页归档：[sources/sites/heracles-humanoid-control.md](../sites/heracles-humanoid-control.md)
- 交叉补强：[人形运动跟踪方法选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)、[扩散运动生成](../../wiki/methods/diffusion-motion-generation.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md)、[BFM 人形基础模型](../../wiki/entities/paper-behavior-foundation-model-humanoid.md)
