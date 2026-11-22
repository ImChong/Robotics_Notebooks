# RoboWild: Can Frontier Models Generalize to Robot Use in the Wild?（论文原文核查）

- **类型：** paper
- **英文标题：** RoboWild: Can Frontier Models Generalize to Robot Use in the Wild?
- **中文标题：** RoboWild：前沿视觉语言模型能否泛化到复杂真实感场景的机器人操作？
- **作者：** XLANG Lab（详见官方仓库 README 中 BibTeX 的完整作者名单）
- **年份：** 2026
- **官方 PDF：** https://huggingface.co/datasets/xlangai/RoboWild/blob/main/RoboWild.pdf
- **项目主页：** https://robowild.xlang.ai/
- **源码：** https://github.com/xlang-ai/RoboWild
- **数据：** https://huggingface.co/datasets/xlangai/RoboWild
- **收录日期：** 2026-11-22
- **核对范围：** 已核对官方项目页 Abstract、ReAct Leaderboard、Results Analysis、开放资源入口；通读官方 GitHub README、评测/任务定义/运行时文档与关键环境包装器源码。**论文 PDF 本次尝试从 Hugging Face blob/resolve 链接获取均失败，未能阅读论文的方法、实验表及附录；以下明确区分项目页公开结论和 PDF 待核项。**
- **通读：** 未完成

## 官方已公开且可验证的关键信息

1. **任务规模与复杂性：** 官网 38 个仿真场景、4,480 件功能/物理可交互物体、341 个任务–本体评测；README 进一步拆为 239 项唯一 Dexmate 任务设置和 34×3 项跨本体复用设置。涉及导航、抓取、功能物体操作、双手协作等。
2. **模型表现：** 项目官网报告 Claude Fable 5.2 weighted success 13.28%（榜单四舍五入 13.3%），其余在官网同协议榜单依次 GPT-6 Astra 11.6%、Gemini 4 Carbon 10.6%、Claude Opus 5.5 6.5%、GPT-6.1 Sol 4.6%；官网标题为 ReAct at highest thinking level。
3. **失败归因：** 官方 Results Analysis 标记 188 个失败运行样本，Grasp Failure 42.6%、Motion/Execution 28.7%、Functional Object Operation 16.0%、Goal/Target Selection Error 6.9%、Other 5.9%。
4. **恢复与安全：** 局部错误恢复率 Interface 78.7%、Re-anchor 45.5%、Contact 25.7%；无关物体扰动 87%，碰撞停止 78%，目标物品滑落/倾倒 75%，均来自官网可见结果页。
5. **协议边界：** GitHub README 指明默认 RGB-only ReAct + react_move、冻结仿真、实体动作工具，调用预算 Short 100 / Long 200。成绩不代表真实家庭部署或通用 VLA 本体控制能力。

## 待 PDF 核查

- 正式论文的模型/本体分组表格、各机器人（尤其 G1）的确切分母与成功数、加权公式及不确定性。
- 深度观察或代码执行的 harness 消融具体幅度、不同接口或动作计划器干预的因果边界。
- 官网 188 个失败分析样本如何选取；安全副作用各项指标分母与定义。
- 附录中场景设计、资产建模和评价器验证详情。

**审慎结论：** 目前可确认“复杂仿真场景中接触执行与失败恢复是显著瓶颈”，但不能据此证明“模型已经完全掌握高层规划”或“真机家务成功率为 13%”。

## 对 wiki 的映射

- [论文详情](../../wiki/entities/paper-robowild.md)
- [官方主页核对](../sites/robowild-xlang-ai.md)
- [官方源码核对](../repos/robowild.md)
