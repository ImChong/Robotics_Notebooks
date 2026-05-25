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
- **一句话说明：** **状态条件扩散中间件**插在高层参考运动与底层物理跟踪策略之间：状态贴近参考时近似恒等映射保零样本跟踪精度，大偏差时生成类人恢复关键帧轨迹，经闭环重规划喂给通用 tracker，无需显式 tracking/recovery 模式机。

## 摘要级要点

- **痛点：** 通用控制器多把运动控制写成**刚性参考跟踪**；名义条件下有效，但大扰动 / OOD 时盲目最小化即时 kinematic 误差 → 僵硬、非类人、难恢复跌倒。
- **对立面：** 纯生成式 BFM / URL 更自然但难保**时空精确跟踪**；需桥接「精确跟踪」与「生成式恢复」。
- **Heracles 定位：** **中间层**（middleware），非端到端单体策略，也非显式 FSM 切换；以机器人**实时本体状态**条件化扩散过程，**隐式**在 identity map 与 generative synthesizer 间过渡。
- **系统两层：**
  1. **生成中间件：** 低频规划，输入 $(\mathbf{p}_t, \mathbf{m}_t)$，输出短视界关键帧轨迹 $\bm{\tau}_t\in\mathbb{R}^{K\times D}$；
  2. **通用物理跟踪策略 $\pi$：** 高频 MDP，观测含 $\mathbf{p}_t$、由 $\bm{\tau}_t$ 密化后的 $\mathbf{m}'_t$、离散运动嵌入 $\mathbf{z}_d$。
- **残差参数化：** 基线 $\bm{\beta}_{t,k}=\mathbf{p}_t$ 固定，只预测残差 $\mathbf{r}_t$，$\bm{\tau}_t=\bm{\beta}_t+\mathbf{r}_t$；近参考时残差≈0 → 近似透传命令。
- **训练：** **Conditional Flow Matching**（线性插值路径 + 速度场 MSE）；AdaLN-Transformer 速度网络；首 token inpainting 锚定当前状态；推理用 **directional warm start**（向 $\mathbf{m}_t$ 线性插值初值 + 部分噪声，SDEdit 风格）减少 ODE 步数。
- **部署：** 每 $N_{\mathrm{exec}}$ 步重规划；关键帧经三次样条（关节）+ slerp（根朝向）密化为 tracker 频率；形成 **closed-loop tracking–generation**。
- **Tracker 增强：** 论文同时改进底层跟踪器与整体框架（FSQ 量化 motion token、重建与动作预测头等，见正文 §3.3）；真机实验报告极端扰动下**涌现式类人恢复**与运动泛化。

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
