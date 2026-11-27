# STARS: extending interactive task learning with large language models

- **作者：** James R. Kirk, Robert E. Wray, John E. Laird
- **机构：** Center for Integrated Cognition, IQM Research Institute（Ann Arbor, MI, United States）
- **版本 / 载体：** *Frontiers in Computer Science*（Sec. Human-Media Interaction）, Volume 8, 2026, Article 1848830；DOI: `10.3389/fcomp.2026.1848830`（Received 2026-04-06；Revised 2026-06-09；Accepted 2026-07-02；Published 2026-08-04；扩展自作者 AAAI 2024 工作 Kirk et al., 2024）
- **原文：** [Frontiers in Computer Science 原文全文](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1848830/full)
- **项目页 / 代码：** 已开源（核心实验代码、仿真环境与分析脚本发布于官方 GitHub 仓库 [Center-for-Integrated-Cognition/STARS](https://github.com/Center-for-Integrated-Cognition/STARS)，正文 Footnote 1 明确指向该仓库；详见 [STARS 官方代码仓归档](../repos/paper-agentic-survey-doi-ddc6fc88fa219f08.md)）
- **核对范围：** 通读全文 1–6 节（Introduction、Background 2.1–2.3、Preliminary investigation 3.1–3.5 及 Algorithm 1、STARS 方法与实验 4.1–4.3、Discussion 5.1–5.3、Conclusions 6、Statements、Footnotes 1–14 及 References 1–74），完整核对 Table 1–Table 8 与 Figure 1–Figure 14 的文字及数据描述，并交叉核对官方代码仓 `ITL-agent/`、`mobile-sim/` 与 `DataAnalysisScripts/` 结构。
- **通读：** 已完成

## 原文核对笔记

- **研究动机与核心矛盾：** 交互式任务学习（ITL）基于 Soar 认知架构实现单样本（one-shot）在线任务学习，并支持知识迁移与用户定制，但过度依赖人类教员（instructor）提供精细且高频的目标与动作指令；直接引入大语言模型（LLM）虽具备广博常识，却面临四大具身化挑战：在线实时学习（不可针对新任务预微调）、多知识源融合、单样本高效学习、可用且可靠的接地输出。作者在第 2.3 节界定了具身 LLM 响应的四个要求：语言可解析（interpretable）、环境可接地（situated）、匹配本体与可供性（embodiment/affordance）——前三者合称 **viable（可执行）**——以及符合特定用户偏好（**situationally relevant，情境相关**）。
- **预研基线（TBP，第 3 节）：** 基于 Soar 构建的移动单臂机器人 ITL 智能体（Rosie），在 APRIL MAGIC 仿真办公室厨房（30 个待整理物品 + 5 处须关闭的门/抽屉 = 35 条目标断言）上评测 GPT-3（`text-davinci-002`）模板化提示（Template-Based Prompting, TBP）。以温度 0/0.9/1.0 采样 3 个去重候选并按平均对数概率排序：
  - 仅依赖 `Search d<=4 + LLM` 时，无监督完成率仅 `54.3%`（大量目标不可执行，如关闭无盖垃圾桶、将空瓶放入冰箱）；
  - `Inst+Search d<=4 + LLM` 结合人类监督（yes/no 确认及兜底描述）恢复 `100.0%` 完成率，且将教员总词数从纯指令 `757` 词（`Inst+Search d<=4` 为 `603` 词）降至 `360` 词，但 GPT-3 目标响应被人类接受的相关率仅 `30.2%`（动作首查正确率达 `96%`，主要瓶颈在于目标描述）。
- **STARS 方法设计（第 4.1 节，Search Tree, Analyze, Repair, and Select）：**
  1. **Search Tree (ST)：** 用单次提示上的束搜索（beam search）替代升温重采样，在 GPT-3 completions API 下，当生成 token 的对数概率 `< 90%` 时分叉，忽略 `< 5%` 分支，每处最多扩展 4 个候选，最大递归深度限制为 3，高效获得多样且无重复的高概率候选树。
  2. **Analyze and Repair (AR)：** 智能体在内部认知模拟（mental simulation）中复用原生 NLP 解析器、情景接地与语义/感知记忆检查三类不匹配——语言未知词/不可解析（Language）、环境物体指代无法接地（Situation）、本体可供性冲突（Embodiment/Affordance，例如“dish rack”不具备 `grabbable` 属性）。随后基于诊断结果构造定向修复模板（如 `"No. Cannot see a cabinet."` / `"No. Rack is not grabbable."`），触发 ST 重新生成修复后的候选树。
  3. **Selection (S) 与 User Oversight (O)：** 将所有通过 AR 筛选的可行候选编号后交由 GPT-4（chat endpoint）选出最合理目标索引；在 `STARS+O` 条件下由人类教员以 yes/no 确认（最多审查 5 个候选，拒绝则重选或由人补充）以捕获个体偏好。成功执行后，Soar 通过回溯分析（retrospective analysis）将因果依赖编译为长期程序性策略知识，后续同类任务无需再查 LLM 或搜索。
- **主实验结果（第 4.2–4.3 节，Table 7–8）：** 在三个仿真任务（Tidy kitchen 35 物体、Store groceries 15 物体、Organize office 12 物体；共 58 个独特物体类型）上对比 `TBP`、`TBP+O`、`STARS`、`STARS+O` 与二次复现 `Trial #2`：
  - **无监督完成率：** `TBP` 在厨房/杂货/办公室分别为 `52.5%` / `66.7%` / `35.7%`；`STARS` 提升至 `77.5%` / `94.4%` / `92.9%`（10 次独立运行均值与标准差分别为 `77.5%±2.04%`、`93.89%±1.76%`、`92.14%±2.26%`）；在厨房任务系统提示中补充脏污/空瓶/干货常识上下文（`STARS*`）后，无监督完成率进一步升至 `92.5%`。
  - **监督开销与响应质量：** `STARS+O` 在三项任务均达 `100.0%` 完成率，教员输入词数较 `TBP+O` 显著下降（厨房 `403 → 127`、杂货 `92 → 44`、办公室 `184 → 60`），人类 yes/no 接受率由 `33% / 62% / 18%` 跃升至 `69% / 94% / 73%`；且 `STARS` 在厨房任务中使 `100%` 物体（35/35）至少获得一个情境相关候选（`TBP` 仅 15/35），最终使用响应 `100%` 达到“至少合理（reasonable）”水平；Token 消耗约为 `TBP` 的 3–6 倍（厨房 `41K → 139K`、杂货 `17K → 62K`、办公室 `13K → 76K`）。
  - **Trial #2 零开销迁移：** 经 `STARS+O` 学习并编译长期策略后，二次下达任务（1 条指令、2 个词）在三项任务上均以 `0` 次 LLM 查询、`0` token 消耗实现 `100.0%` 完成。
- **AI2-THOR 扩展探索（第 5.2 节）：** 将 STARS 拓展至两类新知识（以演示案例验证，尚未做大规模定量评测）：（1）**动作与递归子任务学习**（如 `"Cook a carrot"` 遇未知动词 `"Rinse"` 时递归调用 STARS 学习洗胡萝卜，并由 AR 自动修正 `"stoveburner"` → `"stoveknob"`、`"onto metal pan"` → `"into metal pan"`，教员补充关闭炉灶/水龙头以防 post-completion error，学成后可迁移至 `"cook a potato"`）；（2）**类别概念接地定义**（在无人类监督下学习 `"all recyclables are in the recycling bin"` 等全称规则，通过比对可见物体集合拒绝如 `"metal -> recyclable"` 等误匹配金属锅的过泛化定义，仅保留存在遗漏关柜门等后置误差的局限）。
- **对 wiki 的映射：** [STARS 论文实体页](../../wiki/entities/paper-agentic-survey-doi-ddc6fc88fa219f08.md)
