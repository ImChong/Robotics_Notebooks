# Safe Task Planning with Long-Term Graph Memory for Embodied Agents

- **作者：** Siyuan Li、Taiyan Lang、Aoqi Yan、Jia Yu、Feifan Liu、Yihan Du、Yu Zheng、Xun Wang、Peng Liu
- **机构：** 哈尔滨工业大学计算学部（Faculty of Computing, Harbin Institute of Technology）、新加坡科技设计大学（Engineering Systems and Design, Singapore University of Technology and Design）、Academy of CASIC（Intelligent Science and Technology）
- **版本 / 载体：** arXiv:2609.08444v1 [cs.RO]（2026-09-08 提交；arXiv 元数据注释与项目页标注 CoRL 2026）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2609.08444)、[arXiv 摘要页](https://arxiv.org/abs/2609.08444)
- **项目页 / 代码：** <https://sites.google.com/view/safemem>（项目页，含论文链接、代码链接与演示视频）；<https://github.com/lty759/SafeMem>（**已开源**：MIT License，含仿真评测主入口 `og_ego_prim/`、增量场景图工作进程 `scene_graph/`、四类风险任务列表 `entrypoints/` 与风险预测提示词 `configs/prompts/risk.txt`；Franka Panda 真机控制脚本未见于公开目录清单）
- **核对范围：** 全文通读 arXiv HTML v1（`paper_text.txt` 第 1–1471 行），覆盖 Abstract、§1–6、Table 1–13、Figure 1–5 题注与正文解读，以及 Appendix A（A.1–A.8 边定义、完整提示词、3D 包围盒几何与可计算关系规则、冲突优先级、超参数、Algorithm 1 伪代码）和 Appendix B（B.1–B.2 仿真与 Franka Panda 真机实验设置、低层技能集、Token 与延迟统计）；并核对项目页文本快照 `project_page.txt`。
- **通读：** 已完成

## 原文核对笔记

### 1. 问题设定与动机（§1, §3）

1. **部分可观测下的物理风险盲区：** 现有 LLM/VLM 安全任务规划方法（如安全提示词、CoT 安全提醒、SaP 认知提示）主要针对当前视野内可见的危险源；但在以自我为中心（egocentric）视角的部分可观测环境下，先前见过的危险物（如插线板、易燃物）会在后续操作中被遮挡或移出视野，导致规划器生成高危动作（如在插线板旁给盆栽浇水）。
2. **POMDP 形式化与双目标评测：** 将任务规划建模为 $\mathcal{M} = \langle \mathcal{O}, \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{L} \rangle$，每步接收单视角 egocentric RGB-D 观测 $o_t$，从预定义可执行技能集 $\Pi$ 输出高层动作 $a_t$。遵循 IS-Bench 使用 BDDL 谓词定义任务目标 $\mathcal{G}_{task}$ 与安全目标 $\mathcal{G}_{safe}$，以成功率 SR（仅满足 $\mathcal{G}_{task}$）和安全成功率 SSR（同时满足 $\mathcal{G}_{task}$ 与 $\mathcal{G}_{safe}$）共同评估。

### 2. 核心框架：SafeMem（§4, Appendix A）

1. **长期语义图记忆 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$（§4.1）：**
   - **节点与闭集关系：** 节点 $v_k = \{c_k, id_k, v_k^b, m_k\}$ 记录类别、跨时间步持久实例 ID、2D 包围盒与分割掩码；对涉及可计算空间关系的节点，利用深度图 $o_t^D$ 和相机内参 $K$ 反投影点云 $\mathbf{p}_j = d_j K^{-1}[x_j, y_j, 1]^\top$ 构建轴对齐 3D 包围盒 $B_k$。关系集合 $R$ 划分为几何可计算关系 $R^{cal} = \{\text{ON}, \text{IN}, \text{ABOVE}, \text{NEAR}, \text{ATTACHED TO}\}$ 与语义关系 $R^{incal} = \{\text{PART OF}, \text{HOLDING}\}$。
   - **帧级图初始化（§4.1.1, Appendix A.3–A.7）：** 用 Gemini 2.5 Flash 从 $o_t^{RGB}$ 在闭集物体词表与关系词表约束下生成初始 JSON 图 $\tilde{\mathcal{G}}_t$；并行用 SAM 2.0 网格采样生成掩码集 $M$；通过 IoU 最大匹配将节点对齐到 SAM 掩码，滤除 $\text{IoU} < \epsilon\ (0.10)$ 的误匹配及同掩码重复节点；再用 3D 包围盒规则和优先级 $\text{IN} > \text{ON} > \text{ATTACHED\_TO} > \text{ABOVE} > \text{NEAR}$ 纠正 LMM 空间关系边。
   - **跨时间步图更新 $\mathcal{G}_t = \text{Merge}(\mathcal{G}_{t-1}, a_{t-1}, \mathcal{G}_t^{new})$（§4.1.2）：** 基于上一帧与当前帧掩码 IoU 做二分匹配（阈值 $\tau$），匹配节点沿用持久 ID 并刷新几何与标签；未匹配的新节点分配新 ID；**暂时未匹配的历史节点予以保留**以应对遮挡与出视野；对不可见物体保留历史边，但裁剪上一动作 $a_{t-1}$ 交互对象与不可见物体之间的旧边，并按 $a_{t-1}$ 语义追加新边（如放入容器后补充 `IN`）。
2. **安全感知规划与保守度调制重规划（§4.2, Appendix A.8）：**
   - **风险预测器（GPT-4o）：** 输入剔除掩码和坐标后的精简图记忆（仅保留物体名称/ID 与显式关系三元组）和候选动作 $\hat{a}_t$。提示词严格约束只能依据图内显式边做单步关联推理，禁止仅因同在一张桌面就臆测 `NEAR` 或做多跳传递推理；若存在风险则输出 `REPLAN` 信号与引用具体图关系的解释 `reason`。
   - **VLM 规划器与保守因子 $\eta$：** 当收到 `REPLAN` 时，以概率 $\eta \in [0, 1]$（默认 $\eta = 0.8$）将 `reason` 反馈给 GPT-4o 规划器重新生成候选动作，循环直至通过安全检查或触发随机放行，避免风险预测器与规划器之间出现死锁。

### 3. 实验结论与量化数据（§5, Appendix B）

- **仿真对比（IS-Bench，17 个任务覆盖触电/火灾/跌落/食物污染 4 类风险，Table 1–2）：**
  - 部分可观测设定下，SafeMem 平均 SR / SSR 为 **0.79 / 0.59**，显著高于 SaP（0.79 / 0.13）、CoT（0.83 / 0.17）、Base（0.59 / 0.00）和无图记忆变体 SafeMem w/o GM（0.75 / 0.21）。
  - 全可观测设定下，SafeMem 平均 SR / SSR 为 **0.88 / 0.67**，仍领先 SafeMem w/o GM（0.92 / 0.42）、CoT（0.88 / 0.21）、Base（0.79 / 0.21）和 SaP（0.88 / 0.13）。
- **真机对比（7-DoF Franka Panda + RealSense D405，4 类任务各 10 次试验，Table 3）：**
  - SafeMem 在四类任务上均取得 **1.0 SR / 1.0 SSR**，而所有基线方法在保持较高 SR（0.6–1.0）的同时 SSR 几乎全部跌至 **0.0–0.1**（SafeMem w/o GM 仅在触电任务达 0.3 SSR，其余为 0.0）。
- **消融与开销（Table 4–7, Table 11）：**
  - 记忆形式对比（部分可观测）：图记忆 SSR **0.59** > RAG（0.38）> 无边物体列表 object-list（0.17）> 全历史图像 full（0.13）。
  - 组件消融：移除不可见对象保留机制 SSR 跌至 **0.21**；移除动作感知边更新 SSR 降至 **0.38**。
  - 保守因子 $\eta=1$ 可将安全违规率从 $\eta=0.8$ 的 **0.35 (6/17)** 降至 **0.18 (3/17)**（SSR 升至 **0.67**），但会增加重规划延迟；在无危险源任务中（Table 7）SafeMem 平均 SR 为 **0.82**，未造成明显过度保守。
  - 单步平均开销（Table 11）：Base 为 3979.77 tokens / 10.33 s，SafeMem w/o GM 为 6101.48 tokens / 12.43 s，SafeMem 为 9425.40 tokens / 17.62 s。

对 wiki 的映射：[SafeMem 论文实体](../../wiki/entities/paper-agentic-2609-08444.md)。
