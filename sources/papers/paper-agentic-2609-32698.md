# SEES: A Self-Evolving Embodied System via Failure-Guided VLA Policy Adaptation

- **作者：** Ziwen Li, Hanlue Zhang, Zhenyang Ren, Tianyu Huang, Runqi Lin, Haoyu Wang, Zhengqing Gao, Yandong Guo, Fakhri Karray, Tongliang Liu, Chris Russell, Mingming Gong
- **机构：** Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), University of Sydney, University of Oxford, AI^2 Robotics, University of Waterloo, University of Melbourne
- **版本 / 载体：** arXiv:2609.32698v1 [cs.RO]（2026-09-26 提交，arXiv HTML 全文）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2609.32698) / [arXiv HTML 全文](https://arxiv.org/html/2609.32698)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv 原文及元信息未提供公开代码仓库或独立项目主页链接；正文说明训练管线基于 RLinf 实现，附录 A 提及补充提示文件 `task_decomposition.md`）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1 Introduction、§2 Related Work、§3.1–3.3 Method、§4.1–4.5 Experiments 与 Tables 1–4、§5 Real-World Feasibility and Future Work、§6 Conclusion、Appendix A–E 与 Tables 5–7）
- **通读：** 已完成

## 原文核对笔记

- **核心问题：** 视觉-语言-动作（VLA）策略在短程操作上泛化较好，但在多阶段长程任务上可靠性低，原因是大规模示教数据偏向单阶段任务，少数薄弱原子技能会在多条长程任务中反复造成失败；传统专家干预再收集 SFT 示教成本高且部署后不可持续。
- **系统闭环（§3）：**
  1. **任务分解与技能族路由（§3.1 & Appendix A）：** 使用冻结 VLM（Gemma 4 31B IT，贪心解码）将长程指令与三视角初始图像分解为有序原子任务序列 $\mathcal{A}_c$（64 个场景测得 63/64 = 98.44% 整计划一致率、100% 合法 JSON），语言智能体将每个原子任务映射至 11 个预定义语义技能族之一 $g(i)$，同一技能族共享一个动作头 LoRA 适配器 $\phi_f$。
  2. **终止头与失败归因（§3.1 & Appendix B）：** 在冻结 VLA 动作 token 特征上附加轻量两输出头（LayerNorm + 512 维 GELU + 2 输出：终止概率与进度），头参数先用 $\mathcal{L}_{\mathrm{head}} = 0.10\,\mathcal{L}_{\mathrm{term}}^{\mathrm{SFT}} + 0.05\,\mathcal{L}_{\mathrm{progress}}$ 离线训练；复合任务执行时连续 3 步预测终止概率 $\ge 0.95$ 即判定子任务完成并切换下一阶段。失败轨迹归因到首个阻塞继续执行的原子阶段（Eq. 1: $b_c = \arg\max_{i \in \mathcal{A}_c} F_{c,i}$），每条复合任务每轮选 1 个最高频瓶颈。表 5 审计（100 条复合轨迹、243 次原子尝试）显示该终止头在结果 Macro-F1（94.0% vs. 59.9%）与首因失败归因准确率（92.0% vs. 45.0%）上显著优于 Qwen3-VL-32B-Instruct 轮询判定。
  3. **瓶颈导向仿真 RL 任务构建（§3.2 & Appendix C.1–C.2）：** 从复合执行轨迹中采集瓶颈阶段开始前的完整仿真与任务依赖状态；由 LLM（GPT-5.6-Sol）根据指令、对象标识与仿真查询接口文档生成布尔成功谓词 $C_i(s)$（表 6 在 900 条 Atomic-Seen 轨迹上帧一致率 97.8%、回合一致率 95.9%），过滤不可恢复或已满足状态的样本构成 $\mathcal{I}_i$（Eq. 2），并以连续 3 步满足 $C_i$ 触发的锁存差分稀疏奖励 $r_{i,t} = S_{i,t} - S_{i,t-1}$ 训练（Eq. 3）。
  4. **技能族共享 LoRA 演化与验证注册（§3.3 & Appendix C.3–C.5）：** 按 $w_{i,k} = \sqrt{N_{i,k}}\,2^{-(k - k_i^{\mathrm{first}})}$ 采样当轮瓶颈及历史同族任务（Eq. 9），在 RLinf 上使用 PPO 联合更新动作头 LoRA（RoboCasa365 为 rank=64, $\alpha=128$；LIBERO 为 rank=8, $\alpha=8$）及价值/终止头（Eq. 10: $\mathcal{L}_{\mathrm{RL}} = -J_f^{\mathrm{PPO}} + \mathcal{L}_V + 0.05\,\mathcal{L}_{\mathrm{term}}^{\mathrm{RL}}$），冻结 VLA 主干。仅当验证集上瓶颈任务等权平均成功率增量 $\Delta_T^{(j^*)} > 0$ 时才注册新检查点（Eq. 7）。
- **主要实验结论（§4）：**
  - **RoboCasa365（表 1）：** 三轮演化后，GR00T N1.5 的 Atomic / Composite-Seen / Composite-Unseen 成功率从基线 47.2% / 10.2% / 3.5% 提升至 55.7% / 13.1% / 6.1%；RLDX-1 从 66.4% / 20.4% / 6.0% 提升至 70.3% / 21.8% / 7.1%。执行态未适应前（execution-only）因部分分解子任务在单阶段 SFT 数据中缺失而出现 Composite-Seen 暂降（GR00T 10.2%→9.5%，RLDX-1 20.4%→17.6%），经演化后超越原始基线。
  - **LIBERO 四套件（表 2 & 表 7）：** 跨套件平均成功率由 53.10% 升至 65.40%，未参与 PPO 训练的同族兄弟任务在四个套件均获正迁移（如 Object 套件兄弟任务 82.22%→93.33%，全套件 10 个任务全部提升）。
  - **消融验证（表 3–4）：** 在 5 个 pick-and-place 任务上，共享家族 LoRA 相比独立原子 LoRA 在瓶颈任务（74.0% vs. 70.7%）与未训练兄弟任务（80.0% vs. 72.0%）均更高；在 `ScrubCuttingBoard` 上，瓶颈选择两轮后全任务成功率达 32/50（64%），高于随机（28%）与高成功率优先（34%）。
- **对 wiki 的映射：** [SEES: A Self-Evolving Embodied System via Failure-Guided VLA Policy Adaptation](../../wiki/entities/paper-agentic-2609-32698.md)
