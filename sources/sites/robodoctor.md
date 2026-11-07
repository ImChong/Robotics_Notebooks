# RoboDoctor 官方项目主页

> 来源归档（site；2026-11-07 重新核查）

- **标题：** RoboDoctor: Training-Free Recursive Self-Improvement for Robot Policies
- **类型：** site
- **官方项目页：** <https://robodoctor-rsi.github.io/>
- **论文：** <https://arxiv.org/abs/2610.35604>
- **技术报告：** <https://robodoctor-rsi.github.io/tech_report.pdf>
- **代码：** <https://github.com/THUSI-Lab/RoboDoctor>
- **权重：** <https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robodojo>
- **入库日期：** 2026-10-28，2026-11-07 更新
- **核对范围：** 项目页顶部 Paper/arXiv/Code 按钮、方法/评测与倒水演示、论文 Figure 4 对应片段、正式 GitHub 仓库 README 和可执行文件入口。
- **通读：** 已完成

## 资料及开放状态

官网展示控制分配、选择性修正和经验管理，倒水演示标记 `pour_liquid_into_cup · iter3 · tick 0`。2026-11-07 官网的 **Code** 入口已指向 [THUSI-Lab/RoboDoctor](https://github.com/THUSI-Lab/RoboDoctor)，仓库包含 Python 控制器、Shell launchers、测试和部署说明。PrismWAM 权重另在 Hugging Face 独立提供；不能把它当作 RoboDoctor 自带训练权重。

早期源文件曾把 Hugging Face **PrismWAM 模型权重路径**误作 RoboDoctor 官方项目页，此处已按官网更正。之前 `RoboDoctor-Preview` 在 2026-10-22 的空仓库结论属于**历史状态**，不能用于描述目前开源程度。

## 证据边界

- 官网视频是具体回放证据，不等于所有原生任务成功。
- 汇总分数 49.74→57.10 来自筛选后的本地仿真归档；跨方法评测覆盖和交互预算不严格一致。
- 主要方法、局限与全任务表见 [原论文归档](../papers/robodoctor_arxiv_2610_35604.md)。

## Wiki

- [RoboDoctor 唯一实体节点](../../wiki/entities/paper-robodoctor.md)
- [正式源码与运行时序核对](../repos/robodoctor.md)
