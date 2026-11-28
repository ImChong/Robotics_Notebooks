# Contact Harness: Enabling VLM Agents to Perform Contact-Rich Manipulation

- **论文：** <https://arxiv.org/abs/2611.29377v1>
- **PDF：** <https://arxiv.org/pdf/2611.29377v1>
- **作者：** Shilin Shan、Yuxuan Hu、Chuhao Zhou、Yimeng Dong、Bohan Hou、Jiaqi Bai、Boyu Ma、Jiahua Dong、Jianfei Yang
- **机构：** 南洋理工大学（Nanyang Technological University），MARS Lab
- **版本：** v1，2026-11-22
- **核对范围：** PDF 正文 §1–5 及附录 A–I，重点核对多模态与柔顺控制、真机与仿真实验、种子和力规则、失败案例及提示词；对照发布的代码与数据。
- **通读：** 已完成
- **项目页：** <https://ntumars.github.io/ContactHarness/>
- **源码：** <https://github.com/NTUMARS/Contact-Harness>
- **数据：** <https://huggingface.co/datasets/ntumars/Contact-4>

1. 文字技能定义各阶段需要的证据；模型处理慢速决策，快环使用阻抗及力反馈调参考位姿。该方法没有用示范微调，基准的策略对照有使用示范。
2. 表 1 的 90.7% 是 Rokae 上 15 类连接器各 10 次的完全插入结果。表 3 的 96 秒属于快速模式加记忆的 IEC C13 设置，不能代替全量平均。
3. 表 4 的六项仿真按任务成功报 96.7%；表 5 的四项接触消融另加力违规判据。隐藏智能体力观测不等于关闭底层力反馈。
4. 附录 H 提醒远程服务延迟、遮挡、传感歧义与高 token 用量；不能从仿真成功推出安全部署。

**对应 wiki：** [Contact Harness](../../wiki/entities/paper-contact-harness.md)
