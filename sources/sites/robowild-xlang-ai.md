# RoboWild 官方项目主页（XLANG Lab）

- **类型：** site
- **项目：** RoboWild: Can Frontier Models Generalize to Robot Use in the Wild?
- **官网：** https://robowild.xlang.ai/
- **论文：** https://huggingface.co/datasets/xlangai/RoboWild/blob/main/RoboWild.pdf
- **代码：** https://github.com/xlang-ai/RoboWild
- **数据集：** https://huggingface.co/datasets/xlangai/RoboWild
- **核对日期：** 2026-11-22
- **核对范围：** 官网 Abstract、Scenes and Tasks、ReAct Leaderboard、Results Analysis（失败分类、恢复能力、RoboDojo 对照与安全副作用）、项目资源链接。数据来自官网展示，不代表自行重跑。
- **通读：** 已完成

## 官网核对记录

1. **基准规模：** 38 个布置完整的仿真场景、4,480 件具有物理与功能交互的物体、341 组机器人任务评测、4 种机器人本体；场景包括家庭、工作间和实验室，覆盖关节机构、可变形材料、粒子集合以及 60 多种物体交互行为。
2. **榜单协议：** 官网显示 ReAct、模型最高 thinking level 的 **weighted success**。Claude Fable 5.2 13.3%（官网摘要给出 13.28%）、GPT-6 Astra 11.6%、Gemini 4 Carbon 10.6%、Claude Opus 5.5 6.5%、GPT-6.1 Sol 4.6%。这是同站点特定版本/协议榜单，不能解释为真机成功率。
3. **失败分析样本：** 官网注明来自 **188 个失败运行**。Grasp Failure 42.6%；Motion or Execution Failure 28.7%；Functional Object Operation Failure 16.0%；Goal or Target Selection Error 6.9%；Other Failure 5.9%。总和因四舍五入可能与 100% 有偏差。
4. **局部恢复率定义：** 某次恢复片段是否消除了眼前错误（不等于整任务恢复成功）。Interface 78.7%、Re-anchor 45.5%、Contact 25.7%。
5. **安全副作用：** 在官网统计口径的运行中，无关物体被扰动 87%，collision stops 78%，目标物体掉落/倾倒 75%。未获得完整论文时不推断各项分母是否完全一致。
6. **跨基准比较：** 官网仅对 Claude Opus 5.5、100 tool calls 口径报告 RoboDojo 50.00% / RoboWild 4.81%；仍须先看任务集和接口差别，不能直接推导真实 sim-to-real 降幅。
7. **开源状态：** 官方直接给出 GitHub 仿真仓和 Hugging Face 资产数据；README 表明另有 RoboWild-Harness 负责代理运行与运动规划；不是“仅开放论文”。

## 事实边界

- 官网使用“in the wild”指复杂的**仿真场景**，不是在真实家庭中运行机器人。
- 摘要只说模型在人形机器人本体上存在困难；本次未在可访问的一手文本中找到“人形所有任务 0 成功”的精确统计表，因此不将该数字写成已核实事实。
- PDF 链接可定位，但本次工具无法获取 PDF 文件；具体实验表格、分组权重、样本与置信区间须后续核对正文。

## Wiki 映射

- [RoboWild 独立论文详情](../../wiki/entities/paper-robowild.md)
- [RoboWild 源码核查](../repos/robowild.md)
