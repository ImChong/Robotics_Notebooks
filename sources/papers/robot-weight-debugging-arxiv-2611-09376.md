# Can LLMs Debug Robots by Editing Their Weights?｜让语言模型直接稀疏编辑机器人策略权重

- **类型：** paper
- **论文：** https://arxiv.org/abs/2611.09376
- **原文全文：** https://arxiv.org/html/2611.09376v1
- **原文版本：** arXiv v1；2026-11-28 重核
- **作者：** Jusheng Zhang、Keze Wang、Liang Lin
- **机构：** 斯坦福大学（Stanford University）、X-Era AI
- **核对范围：** arXiv v1 HTML 正文与附录；重点复核 §2 配对协议、§3–5 主要与负结果、附录 A/D/F 的误差审计、留出与候选状态，并对照官方发布 README 与公开配对审计脚本。
- **通读：** 已完成
- **代码：** https://github.com/jushengzhang/robot-weight-debugging
- **wiki：** [唯一论文详情](../../wiki/entities/paper-robot-weight-debugging-2611-09376.md)

## 核心摘录与知识映射

1. 使用现有策略的错误指令与模拟评估，LLM 为少量 token embedding 行提出修补方案；编辑是权重级修改而非新增推理提示。 → [方法说明](../../wiki/entities/paper-robot-weight-debugging-2611-09376.md)。
2. 在原基座 checkpoint 上只替换经过验证的若干 token 行；通过配对留出指令评估干预的增益、退化与未触及的词。 → [方法说明](../../wiki/entities/paper-robot-weight-debugging-2611-09376.md)。
3. 发布 pi0.5 embedding patch、验证报告、episode 记录；重点仍是语义/词汇对齐，其他任务和真机泛化需要分开验证。 → [方法说明](../../wiki/entities/paper-robot-weight-debugging-2611-09376.md)。

## 论文实验事实（非本仓复现）

- 作者在 765 条留出改写指令上报告 π0.5 成功率 660/765（86.3%）→ 704/765（92.0%），成对比较增益 48、退化 4。
- 其中涉及修改词行的 335 个输入受影响；不涉及修改词的 430 条在公开审计里成功/失败结果不变。

## 论文局限

- 不能把改变 token 嵌入理解为获得新物理动作技能或解决任意长尾任务。
- 公开的更广泛任务控制、其他策略与真机证据尚不充分；模型基座 checkpoint 需另行获取。
- Apache-2.0 源码，`apply_sparse_rows.py` 只生成核验过的新 `.npy` 覆盖张量，不实际运行机器人策略。

## 2026-11-28 补充核对

- §2：三个 π0.5 语言会话从同一基座分别探索，作者按任务套件/变体哈希切分练习与留出；留出变体与种子固定，不能视作独立的三次环境抽样。
- §4、附录 D：是否保留取决于编辑器最终决定，某些撤回指令没有执行、某些候选仍存在于目录。FastWAM 首轮和 PRO A 的未保留候选结果不能视为成功验证。
- §5：参考指令混入场景和初始状态字段可显著改变 FastWAM 基线；论文独立比较该因素，并未声称所有模型敏感性相同。
- 附录 A/F：二元结果不变与原始轨迹不变不能互换；公开配方核对的 65 个 token 行与实际变动标量数应分别报告。
