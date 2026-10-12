# RSI-Anything: Unlocking Recursive Self-Improvement with World Models

> 来源归档（ingest 关联资料）

- **类型：** arXiv 技术报告
- **arXiv：** <https://arxiv.org/abs/2610.13912>
- **作者：** Taoyong Cui, Pinxian Zeng, Wanghan Xu, Hao Xiao, Shuhan Xue, Qiang Gao, Wanli Ouyang, Pheng Ann Heng, Zhenfei Yin, Yingcheng Wu, Ling Yang
- **提交日期：** 2026-10-09
- **项目页：** <https://phai-labs.com/news/rsi-anything/>
- **代码：** <https://github.com/Gen-Verse/RSI-Anything>（Apache-2.0）
- **一句话说明：** 在 JEPA-Anything 的预测式世界模型基础上，将候选策略/工作流筛选、学习数据筛选、策略更新及按需世界模型适配组织成可验证的迭代改进框架。
- **为何入库：** 为 RSI 从抽象的模型自我改进议题落到跨任务闭环提供案例，也暴露了 world model 预测误差、组件兼容与实验覆盖边界。

## 核心摘录

1. **共享改进环：** Propose → Predict → Select → Evaluate → Learn → Retain。模型先提出候选，再用世界模型预测结果辅助筛选；真实评估反馈进入策略/工作流学习，模型更新需通过组件及兼容性检查。
   - **对 wiki 的映射：** [递归自我改进](../../wiki/concepts/recursive-self-improvement.md)（与完全自主设计后继模型的强定义区分）；[RSI-Anything 实体页](../../wiki/entities/paper-rsi-anything.md)。
2. **代码修复消融：** 在 CodeContests 的 200 个问题、每题 3 个随机种子、固定 ReasonFlux-Coder-7B 及近似在线测试调用预算下，静态验证候选选择成功率为 26.3%，加入预测选择为 29.2%，再加入世界模型适配为 31.2%。
   - **对 wiki 的映射：** RSI-Anything 实体页「评测与结果」；这是特定基准和预算下的消融，不等同于任意代码仓库收益。
3. **GUI 与科学任务：** MiniWoB++ 成功率从 51% 提升至 99%，平均动作数降低 39.4%；LabHorizon F200 L2 五个版本从 21.80 提升至 46.01%；反应优化的 40 次实验报告平均最佳产率 96.15%，对照 Bayesian Optimization 为 82.16%。
   - **对 wiki 的映射：** RSI-Anything 实体页；注明任务协议、实验预算和真实设备执行边界。
4. **证据边界：** 论文称还需跨多轮联合评估确认策略与世界模型能否相互促进；实验室规划结果与真实仪器执行需区分；免疫学假设仍未验证。
   - **对 wiki 的映射：** RSI-Anything 实体页「局限与风险」。

## 对应代码与官网

- [官方仓库归档](../repos/rsi-anything.md)：公共 Apache-2.0 仓库含通用 workflow runner、任务适配器、配置和代码/GUI 流程；复现需额外下载大模型与资产。
- [官方项目页归档](../sites/rsi-anything.md)：包含代码、模型资源和项目方结果概述。
- [RSI-Anything 官方技术稿](https://phai-labs.com/news/rsi-anything/)
