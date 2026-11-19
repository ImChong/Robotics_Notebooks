# Can LLMs Debug Robots by Editing Their Weights?｜让语言模型直接稀疏编辑机器人策略权重

- **类型：** paper
- **论文：** https://arxiv.org/abs/2611.09376
- **原文全文：** https://arxiv.org/html/2611.09376v1
- **原文版本：** arXiv v1；核对日期 2026-11-19
- **核对范围：** arXiv v1 全文、官方 README、edits/pi05-language-c/apply_sparse_rows.py、evidence/audit_new_results.py、公开配对结果协议
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
