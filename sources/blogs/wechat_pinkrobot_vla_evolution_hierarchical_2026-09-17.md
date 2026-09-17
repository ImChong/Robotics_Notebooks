# 前沿 | VLA 演进：从动作 Token 到分层具身智能体

> 来源归档（blog / 微信公众号）

- **标题：** 前沿 | VLA 演进：从动作 Token 到分层具身智能体
- **类型：** blog
- **作者：** PinkRobot（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/w2QP2RXmpA5juqUjyd0tsQ
- **入库日期：** 2026-09-17
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）
- **一句话说明：** 万字综述把 VLA 主线从「动作语言化 / 离散 token 自回归」经 action chunk、Diffusion/Flow、Action Expert，演进到「慢语义 + 快视觉运动 + 更快全身稳定」的多时间尺度层级，并纳入记忆、在线 RL 与跨本体迁移；附 SayCan→π₀.7 时间线与动作表示对照表。
- **沉淀到 wiki：** [`wiki/overview/vla-evolution-lineage.md`](../../wiki/overview/vla-evolution-lineage.md)
- **姊妹篇：** [`wechat_pinkrobot_mimic_evolution_deepmimic_beyondmimic_2026-09-15.md`](wechat_pinkrobot_mimic_evolution_deepmimic_beyondmimic_2026-09-15.md)、[`wechat_pinkrobot_off_on_policy_rl_evolution_2026-09-15.md`](wechat_pinkrobot_off_on_policy_rl_evolution_2026-09-15.md)

## 核心摘录（归纳，非全文）

### 1) VLA 要解决的统一形式

传统 VLM：\(p(\text{text} \mid \text{image}, \text{lang})\)；机器人策略：\(p(a \mid o, s)\)。VLA 统一为 \(p(a \mid o, l, s, m)\)，其中 \(a\) 可为单步或 action chunk，\(m\) 为短期/长期记忆。

### 2) 三次结构迁移（2022—2026）

| 阶段 | 核心变化 | 代表 |
|------|----------|------|
| **① 动作语言化** | 连续动作离散成 token，交给 VLM/LLM 自回归 | RT-1/RT-2、OpenVLA |
| **② 连续生成与专家** | action chunk、Diffusion/Flow、专用 Action Expert | ACT、Diffusion Policy、π₀、RDT-1B、FAST/OFT |
| **③ 多时间尺度层级** | 慢语义 + 快视觉运动 + 更快全身控制；记忆与在线 RL | Helix/Helix 02、GR00T N1.7、Gemini Robotics 2、π\*₀.₆/MEM/RLT/π₀.7 |

### 3) 四类结构（可组合）

1. **单体自回归 VLA** — RT-2、OpenVLA
2. **VLM + 连续 Action Expert** — π₀、CogACT、GR00T N1
3. **分层双/三系统** — Helix（7–9 Hz / 200 Hz）、Helix 02（+1 kHz System 0）
4. **轻量端侧 VLA** — SmolVLA、Gemini Robotics On-Device 2

### 4) 关键分水岭论文（文内主线）

- **SayCan（2022）** — LLM 语义 × 价值函数 affordance；技能库非端到端 VLA，但奠定「慢规划 / 快执行」分工
- **Gato / VIMA** — 统一序列接口与多模态 Prompt
- **RT-1 → RT-2** — 大规模机器人 Transformer → Web 语义 co-fine-tune + 动作 token；瓶颈：量化、AR 延迟、多峰表达
- **ACT / Diffusion Policy** — 非 VLA 但奠定 action chunk 与条件生成式动作头
- **OXE/RT-X / Octo / OpenVLA** — 跨本体数据标准化与开源 VLA 生态
- **π₀** — VLM + Flow Matching Action Expert，语义与运动解耦
- **FAST / OFT / SmolVLA** — 动作 tokenizer、并行 chunk、端侧异步
- **GR00T N1→N1.7** — 工业人形 VLA + Flow DiT + 部署链路（ONNX/TRT）
- **Helix 02** — System 2/1/0 对应规划 / 视觉运动 / WBC 频率层
- **π\*₀.₆ RECAP / MEM / RLT / π₀.7** — 经验驱动后训练、多尺度记忆、RL 残差、steering 条件

### 5) 动作表示机制对照（文内 Table 2 摘要）

| 机制 | 代表 | 优点 | 主要代价 |
|------|------|------|----------|
| 逐维离散 token | RT-2、OpenVLA | 复用 LLM CE 训练 | 量化、AR 延迟 |
| 频域 token | FAST | 高频序列更短 | tokenizer 复杂 |
| 连续并行 chunk | ACT、OFT | 低延迟 | 多峰弱于生成模型 |
| Diffusion | Octo、RDT、CogACT | 多峰连续轨迹 | 多步去噪开销 |
| Flow Matching | π₀、GR00T、SmolVLA | 较少积分步 | ODE 仍有推理成本 |
| RL 残差/修正 | RLT | 在线精修 | 安全探索难 |

### 6) 多时间尺度与经典栈对应

| 频率 | VLA 模块 | 类比传统模块 |
|------|----------|--------------|
| 1–10 Hz | VLM / System 2 | 任务规划、行为树 |
| 10–200 Hz | Action Expert / System 1 | 视觉伺服、局部 MPC |
| 200 Hz–1 kHz+ | System 0 | WBC、阻抗、关节伺服 |

### 7) 六句结论（文内 §18）

1. Gato/SayCan/VIMA — 语言、视觉、动作可进统一序列与任务接口
2. RT-1→RT-2 — 动作作为语言的 VLA 定义
3. OXE/Octo/OpenVLA — 跨本体数据与开源训练
4. ACT/Diffusion→π₀/CogACT — 从单步离散 token 到 chunk + Action Expert
5. FAST/OFT/SmolVLA — tokenizer、解码、异步与端侧效率是成败核心
6. π₀.5/GR00T/Helix 02/Gemini 2/π₀.7 — 多时间尺度 + 跨本体 + 记忆 + 在线学习 + steering 的完整物理智能体

## 对 wiki 的映射

- **新建：** [VLA 演进技术地图](../../wiki/overview/vla-evolution-lineage.md)
- **交叉补强：** [VLA 方法页](../../wiki/methods/vla.md)、[五大具身模型分类](../../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)、[VLA 14 篇阅读地图](../../wiki/overview/vla-wm-reading-roadmap-14-papers-technology-map.md)、[SayCan](../../wiki/methods/saycan.md)、[RT 系列](../../wiki/methods/robotics-transformer-rt-series.md)、[π₀.₇](../../wiki/methods/pi07-policy.md)
