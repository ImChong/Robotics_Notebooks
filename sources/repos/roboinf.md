# RoboInF 官方评测实现 / RoboInF Official Evaluation Code

- **类型：** repo
- **代码：** <https://github.com/SJTU-DeepVisionLab/RoboInF>
- **核验版本：** v1.0.0 已发布；源码核对固定于 main 快照 [af9d8d5](https://github.com/SJTU-DeepVisionLab/RoboInF/tree/af9d8d583885df48b300ae97f1a45bd07d318b6a)，不是标签提交。
- **许可：** 作者贡献 CC BY 4.0；RoboTwin 等第三方组件保留各自许可。
- **核对范围：** 核对项目页、README、CLI、suite / runner / scoring、RoboTwin 接口及策略文档，用公开评测 JSONL 验证输入；未运行仿真或远程推理服务。
- **通读：** 已完成
- **项目页：** <https://sjtu-deepvisionlab.github.io/RoboInF/> · [论文归档](../papers/roboinf_arxiv_2611_30555.md) · [配套数据](../datasets/roboinf-4k.md)

`inspect` 自检不需要模拟器、GPU 或策略权重，实测 `val`、`test`、`test_random` 各为 8 任务 / 115 场景 / 861 用例。`run` 需要兼容 RoboTwin、资产和策略服务；公开的 `pi_05`、`openwam`、`gr00t` 和 `lingbot-va` 是客户端接口，不是检查点或各方法成绩。`score` 消费外部检查器判决，不能用 rollout 动作文件替代判决。官方留出集参考信息、评分服务及项目检查点未随仓库发布。

- **对 wiki 的映射：** [RoboInF](../../wiki/entities/paper-roboinf.md)
