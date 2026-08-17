# symbiosis-robotics.com/research/dpc（Direct Perception Control 项目页）

> 来源归档（ingest · 项目页）

- **标题：** Direct Perception Control Model / SYMBIOSIS Research
- **类型：** site / project-page
- **官方入口（英）：** <https://symbiosis-robotics.com/research/dpc/en/>
- **官方入口（规范 URL）：** <https://symbiosis-robotics.com/research/dpc>
- **中文切换：** 页头 `hreflang="zh-CN"` 指向 `../`（与英文稿同站；抓取时正文仍可能先出英文）
- **机构：** Symbiosis Robotics（页脚 © 2026 SYMBIOSIS；联系 [info@symbioact.com](mailto:info@symbioact.com)）
- **发表：** August 2026 · SYMBIOSIS RESEARCH
- **入库日期：** 2026-08-17
- **一句话说明：** 主张去掉分层感知–控制里的中间运动接口 \(Z_t\)（以冻结 [SONIC](../../wiki/methods/sonic-motion-tracking.md) 解码器为靶），用单一模型把视觉/语言/本体/动作历史直接映射到 G1 可执行关节与手部 PD 目标；配套 Symbiotic Attention 与 DriftDistill 闭环蒸馏；自报统一语料 **15,010 小时**。
- **开源状态（2026-08-17 核查）：** **确认未开源。** 页头/页脚/Resources 无 GitHub、Hugging Face、Zenodo 或数据集下载；无 arXiv/PDF 链接；Citation 标注为 *Symbiosis Robotics Blog*。可复现入口仅联系邮箱。

## 页面公开信息

| 资源 | URL / 结论 |
|------|------------|
| 英文项目页 | <https://symbiosis-robotics.com/research/dpc/en/> |
| 规范引用 URL | <https://symbiosis-robotics.com/research/dpc> |
| 论文 PDF / arXiv | **无**（截至入库日） |
| 代码 | **未列链接** |
| 权重 / 数据集 | **未列链接** |
| 联系 | [info@symbioact.com](mailto:info@symbioact.com) |

页内脚注（侧栏 Related reading，非本页开源声明）：

| # | 类型 | 条目 | URL |
|---|------|------|-----|
| 1 | Project | Figure Helix | <https://www.figure.ai/news/helix> |
| 2 | Article | Gemini Robotics 2 | <https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/> |
| 3 | Paper | ω-0（Li et al.） | <https://arxiv.org/abs/2608.06375> |
| 4 | Paper | MotionWAM（Zheng et al.） | <https://arxiv.org/abs/2606.09215> |
| 5 | Project | Ψ₀（PSI Lab） | <https://psi-lab.ai/Psi0/> |
| 6 | Paper | π0.5 | <https://www.physicalintelligence.company/download/pi05.pdf> |
| 7 | Article | π0.7 | <https://www.physicalintelligence.company/blog/pi07> |
| 8 | Article | Dyna-2 | <https://www.dyna.co/dyna-2> |
| 9 | Article | GEN-0 | <https://generalistai.com/blog/gen-0> |
| 10 | Article | GEN-1 | <https://generalistai.com/blog/gen-1> |

## 公开信息要点（截至入库日）

- **诊断对象：** System 1 Latent Policy → 运动接口 \(Z_t\) → 冻结 System 0 Whole-Body Tracker → 关节 PD 目标。三条失败模式：insufficient motion interface、separate training / coupled inference、low-level action boundary（冻结解码器动作像 \(M_h\)）。
- **对 SONIC 的显式批评：** \(Z_t\) 被当作运动学参考而非控制步关节目标；token 损失在 64 维恒等度量上，动作损失经 \(J_h\in\mathbb{R}^{29\times 64}\) 投影，\(\mathrm{rank}(J_h^\top J_h)\le 29\)；未来视觉分支若只 attend 运动学表示，则 \(\partial L_{\mathrm{future}}/\partial\theta_D=0\)。
- **方法：** Direct Perception Control 去掉中间运动表示；Symbiotic Attention 在共享动作目标下耦合感知与控制表示；异步视觉 + 连续本体反馈闭环。
- **DriftDistill：** Offline BC 初始化 Student → 在线 rollout 访问漂移状态 → Frozen Teacher 给恢复目标 \(a_t^*\) → 最小化 \(L_{\mathrm{rec}}(a_t,a_t^*)\)；Stage 2 混合 visited + demo。
- **数据：** 异构源全部转成 **G1 可执行、时间对齐的关节轨迹**，合计 **15,010 h**（Human Ego 6,781 / Armed Robot 4,024 / Wheeled Humanoid 3,660 / Bipedal Humanoid 545）。三条转换：遥操作标准化、头戴重建重定向、egocentric 手/腕/末端抬到全身。
- **演示（定性、无公开成功率表）：** 移动拾放锥桶；受限姿态全身 loco-manipulation；手–眼–脚协调（右脚油门 + 双手转向）。
- **硬件接口：** 仍用 29 维 PD 目标 \(A_{\mathrm{joint}}=q_{\mathrm{default}}+C_{\mathrm{scale}}A_{\mathrm{raw}}\)（\(\det C_{\mathrm{scale}}\neq 0\)）；主张放开的是冻结解码器流形，不是卸掉限位/力矩饱和。

## 为何值得保留

- 把 2026 年主流「VLA/WAM + 冻结 SONIC/GMT」分层栈写成可证伪的三条信息瓶颈，而不是口号式「端到端更好」。
- 与本库已有 [MotionWAM](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)、[ω-0](../../wiki/entities/paper-omega-0.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md) 形成直接对照：后者把统一 token 当能力，本页把它当能力上限。
- 开源边界清楚：产业演示级博客，**不能**当可复现训练基线。

## 关联资料

- 技术摘录：[`sources/blogs/symbiosis_dpc_direct_perception_control.md`](../blogs/symbiosis_dpc_direct_perception_control.md)
- 升格：[`wiki/entities/paper-dpc.md`](../../wiki/entities/paper-dpc.md)
