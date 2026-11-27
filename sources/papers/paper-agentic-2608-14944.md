# SkillComposer: Learning Reusable Skills for Natural-Language Robot Programming

- **作者：** John Woods, Hasti Seifi
- **机构：** School of Computing and Augmented Intelligence, Arizona State University
- **版本 / 载体：** arXiv:2608.14944v1 [cs.RO]（2026-08-14 提交，8 页，6 图；Submitted to IEEE Humanoids 2026）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2608.14944) / [arXiv Abs](https://arxiv.org/abs/2608.14944)
- **项目页 / 代码：** 未确认开源代码（arXiv:2608.14944v1 正文、图注及参考文献均未公布 GitHub 仓库或独立项目主页 URL）
- **核对范围：** arXiv:2608.14944v1 全文（Abstract、§I–§VI、Table I–II、Fig. 1–6 文字说明与 References）
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **系统定位与交互载体（§I, §III）：** SkillComposer 面向 RCareWorld 仿真环境（Unity 底座，提供 Python/ROS 接口）中的非编程背景用户（如照护人员、临床研究者），通过本地 Gradio 界面（实时仿真画面 + 对话历史 + 技能调用摘要与自然语言解释）实现交互式自然语言机器人编程，并从用户交互历史中持续学习可复用宏技能。
- **机制一：生成-测试闭环（§III-A）：**
  - **初始原语与表示：** 系统初始提供 5 个基础原语函数：`get_info`、`move_to_object`、`move_to_position`、`grasp_object`、`release_object`。Coder LLM 输出受函数 Schema 约束的 JSON 函数调用序列（而非自由 Python 代码）。
  - **两阶段校验：** 先由 Evaluator LLM 根据用户指令评估功能正确性并给出自然语言反馈；通过后再由 Static Validator 校验 JSON 结构、函数名是否在 Schema 集合内、参数合法性及必填参数完整性。
  - **迭代精炼上限：** 任一阶段失败均将反馈追加至 Coder LLM 对话中要求改写，最多迭代 5 轮；超过 5 轮仍未通过则向用户报错，不执行非法程序。通过校验的程序在 RCareWorld 中执行，并送入在线宏学习模块。
- **机制二：面向命令式程序的 Stitch 在线宏学习（§III-B）：**
  - **JSON 转 S-expression：** 将每条已接受的 JSON 程序按函数 Schema 补齐默认参数并转为位置参数形式的符号表达式（s-expression），使用右嵌套二元 `then` 连接动作序列，从而把命令式机器人程序转化为可供库学习算法 Stitch 处理的树结构，并追加至持续增长的语料库。
  - **改进外循环的候选提取与过滤：** 在语料库上运行 Stitch（最多 20 次迭代、最大参数元数 `max arity = 5`）。由于后续抽象可能引用先前抽象且 Stitch 面向函数式语言设计，系统先递归内联所有抽象调用至基础原语、为缺失参数引入新变量，再按四条结构约束过滤：至少包含 2 个基础原语调用、仅将 `then` 用作二元顺序运算符、仅使用已知基础原语、仅在原语参数位置放置变量；经一致变量重命名去重后，重实现外层循环逐轮选取排名最高且合法的抽象重写语料库，最多学到 10 个有效抽象或无新候选为止（每次接受新指令后重新学习）。
  - **文档化与执行展开：** 追踪抽象变量对应的原语参数推断类型，连同宏体及历史上下文例句交给 Documentation LLM 生成可读技能名、功能描述和参数说明，写入统一函数 Schema 库供 Coder/Evaluator LLM 调用；实际执行前再将宏调用展开回基础原语序列以保持与仿真器兼容。
- **消融实验关键数据（§IV-A, §V-A, Table I, Fig. 5）：**
  - 在 `Objects`（基础操作、空间推理、多步动作、目标规划）和 `Feeding`（人机交互、多步人机交互）两个 RCareWorld 场景共 40 条提示上对比 4 个变体（每组运行 5 次）：`Coder LLM`、`Coder LLM + Eval LLM`、`Coder LLM + Stitch`、`SkillComposer`。
  - 生成-测试闭环是成功率提升的主力（如 `Multi-Step HRI` 成功率从 `0.02 ± 0.06` 升至 `0.98 ± 0.06` / 完整版 `1.00 ± 0.00`；`Goal Planning` 从 `0.08 ± 0.18` 升至 `0.68 ± 0.39` / 完整版 `0.76 ± 0.33`），但单纯加评估器会使耗时增至约 4–5 倍；仅加 Stitch 而无评估器无法显著提高成功率（`Goal Planning` `0.00 ± 0.00`、`Multi-Step Actions` `0.32 ± 0.23`）。
  - 完整 `SkillComposer` 在复杂任务上进一步压缩程序长度并实现高频宏复用：`Goal Planning` 平均程序长度从 `13.07 ± 11.21` 降至 `6.17 ± 4.48`（宏使用率 `86.00 ± 21.91%`）；`Multi-Step HRI` 程序长度从 `10.39 ± 6.05` 降至 `5.50 ± 1.75`（宏使用率 `55.82 ± 34.06%`，平均耗时从 `53.95 ± 11.09 s` 降至 `27.98 ± 11.37 s`）。文中示例学到的宏包括 `move_and_place_object`、`place_object_on_plate` 与 `hand_object_to_user`。不过在 `Spatial Reasoning` 上 `SkillComposer` 成功率为 `0.40 ± 0.42`，低于 `Coder LLM + Eval LLM` 的 `0.76 ± 0.43`。
- **用户研究关键数据（§IV-B, §V-B, Table II, Fig. 6）：**
  - 12 名无或初级编程/机器人经验的被试（3 女 9 男，23–67 岁，IRB 批准，约 75 分钟）在 `Feeding` 场景完成三项任务，与基线 `Coder LLM` 做被试内对照（顺序平衡）：
  - `Rearrange Objects`：成功率 `0.92 ± 0.29` vs. 基线 `0.50 ± 0.52`，提示轮数 `1.9 ± 1.2` vs. `5.8 ± 2.2`，耗时 `3:20 ± 1:20` vs. `6:09 ± 2:57`。
  - `Meal Preparation`：成功率同为 `0.92 ± 0.29`，提示轮数 `3.2 ± 2.6` vs. `8.2 ± 5.6`，耗时 `3:59 ± 1:48` vs. `6:10 ± 3:19`。
  - `Feed Care Recipient`：成功率 `0.67 ± 0.49` vs. 基线 `0.75 ± 0.45`，提示轮数 `3.5 ± 1.6` vs. `6.6 ± 2.7`，耗时 `4:16 ± 1:57` vs. `3:48 ± 1:43`（末端微调动作更适合快速单步交互）。
  - 主观评分：SUS 均值 `72.1 (σ = 20.1)` vs. 基线 `61.4 (σ = 24.6)`（差值 `+10.7`，Cohen's $d = 0.46$，但配对 t 检验所有主观维度均 $p > 0.05$）；12 名被试中有 11 名更偏好 SkillComposer。
- **核查边界与未公布细节：** 原文未写明具体采用的底层 LLM 模型名称/版本号及 API 温度超参；系统仅在 RCareWorld 仿真环境中验证，未在物理机器人真机测试。
- **对 wiki 的映射：** [SkillComposer: Learning Reusable Skills for Natural-Language Robot Programming](../../wiki/entities/paper-agentic-2608-14944.md)
