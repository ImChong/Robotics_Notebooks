# Hypothesis-driven Model Expansion under Uncertainty for Open-World Robot Planning

- **作者：** Anxing Xiao, Hanbo Zhang, Tianrun Hu, David Hsu
- **机构：** School of Computing, National University of Singapore; Smart Systems Institute, National University of Singapore
- **版本 / 载体：** arXiv:2607.06501v1 [cs.RO]（2026-07-07 提交），Accepted to Robotics: Science and Systems (RSS) 2026
- **原文：** https://arxiv.org/html/2607.06501 （arXiv 摘要页：https://arxiv.org/abs/2607.06501 ）
- **项目页 / 代码：** 部分开源（仿真端已开源：https://github.com/threefruits/HUME ；官方仓库 README 明确说明包含 Block Processing World 与 AI2-THOR/ProcTHOR Mobile Manipulation World 两个仿真域及五种实验配置，因系统工程量较大未发布真机 Fetch 与微波炉面板操作演示代码）
- **核对范围：** 已通读 arXiv HTML 全文，包括 Abstract、I Introduction、II Related Work、III Problem Formulation（III-A 至 III-B）、IV Method & Algorithm 1（IV-A 至 IV-F）、V Experiments and Results（V-A 至 V-E）、VI Discussion、VII Conclusion，以及 Appendix A.I–A.IV（真机 Fetch 与 Autolife S2 实验细节、Table I–II、Block World / AI2-THOR 实验设置、A.II 行为与失败案例分析、A.III PDDL 域 Listing 1–6 与完整 Prompt Listing 7–15、A.IV 局限与复杂度分析）；并核对项目 GitHub 仓库页面与 `README.md`、`files_sample`。
- **通读：** 已完成

## 原文核对笔记

- **核心问题与设定：** 针对开放世界机器人任务规划中模型不完备问题（未知物体存在/位置、未知视觉/语义属性、未建模动作效果），将模型扩展显式建模为带不确定性的对象中心假设集合 $\mathcal{H}=\{h_1,\ldots,h_n\}$，通过 Bayes-adaptive MDP 视角将假设验证与目标求解统一在同一个规划循环中。
- **核心方法（HUME）：**
  1. **混合世界模型初始化：** 环境知识用场景图（rooms/locations/objects）表示，动作动力学用 PDDL 域文件定义。
  2. **任务导向假设生成：** 将自然语言指令 $L$ 解析为一阶逻辑目标 $\mathcal{G}$，对比当前模型 $\mathcal{M}_0$ 提取缺失对象 $\mathcal{O}_{\text{miss}}$ 与缺失谓词 $\mathcal{P}_{\text{miss}}$（并由 LLM 分类为固定属性谓词 $\mathcal{P}_{\text{attr}}$ 与动作效果谓词 $\mathcal{P}_{\text{eff}}$），随后由 $\Phi_{\text{LLM}}$ 结合执行历史 $C_{\text{hist}}$ 生成三类假设（`object_existence`、`object_attribute`、`action_effect`），每条假设附带依赖条件 `condition` 与验证前提 `verification_condition`。
  3. **不确定性感知规划与全结果决定化（All-Outcomes Determinization）：** 为每个假设引入三值信念 $b(v(h))\in\{\texttt{True},\texttt{False},\texttt{Unknown}\}$，生成验证动作 $a_{\text{verify}}^h$ 并采取乐观分支剪枝（保留正结果 $a_{\text{verify}}^{h+}$、忽略负结果分支而在执行失败时重规划）。通过条件动作拆分（正常动作代价 1 vs 依赖未验证假设的动作代价 10）引导 Fast Downward（微波炉数值规划改用 ENHSP + Unified-Planning）优先安排验证。
  4. **闭环验证与模型更新：** 执行验证动作时，物体存在性通过目标检测器（真机采用 OWLv2 + SAM + RGB-D 反投影）重构的场景图比对验证；细粒度属性或动作效果由 VLM（真机手持物体旋转腕部采集 4 个视角；微波炉面板用 `qwen3-vl-235b-a22b-instruct` 识别按钮并用连续帧验证数值增量）问答判定。被否决的假设写入 $C_{\text{hist}}$ 并触发重新假设生成与重规划。
- **评测与核对细节：**
  - 六组对比配置覆盖两维度（表示方式：无模型扩展 / 确定性模型扩展 / 不确定性感知模型扩展；推理器：PDDL 形式规划器 / LLM 规划器），基础大模型统一使用 `gpt-4.1-2025-04-14`。
  - 仿真 1（Block Processing World）：3–8 个积木、3–8 个区域处理器（10 种厨房效果中随机遮盖一半处理器名称），每组随机采样 10 次。仿真 2（AI2-THOR + ProcTHOR）：8 个任务（T1–T8），在 5 个平均含 52 个物体、34 个位置的房屋中各测 3 次。
  - 真机 1（Fetch 移动操作）：5 个长程任务（T1–T5，每任务 3 次），低层抓取采用 VAMP + 逆可达性采样 + 行为树（BTs），开/关门技能用约 50 条 VR 遥操示教微调 $\pi_{0.5}$。
  - 真机 2（Autolife S2 微波炉面板操作）：利用 ENHSP 求解数值规划任务“解冻食物 40 秒”，机器人坐下对齐视角，先假设 `+` 按钮增量 10 秒，执行后通过两帧比对发现实为 5 秒并更新模型完成任务。
  - 注：arXiv HTML 原文中仿真与真机对比柱状图/折线图（Fig. 6–8, 10, 29, 30）以图片形式嵌入，未在文本中列出原始数值表；正文与附录明确给出效率与开销数据：开放世界 Blocks World 中重规划触发次数降低约 38.5%，每轮生成/更新耗时 7.6s + 规划耗时 0.25s（直接 LLM 规划为 7.0s），真机全流程中规划时间平均仅占总任务时间的 7.6%。
- **对 wiki 的映射：** [Hypothesis-driven Model Expansion under Uncertainty for Open-World Robot Planning (HUME)](../../wiki/entities/paper-agentic-2607-06501.md)
