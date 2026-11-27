# Stanford CS329Z course code · 公开代码核对

- **官方组织：** https://github.com/cs329z
- **项目页：** https://cs329z.stanford.edu/
- **课程归档：** [CS 329Z](../courses/cs329z-engineering-ai-agents.md)
- **核对日期：** 2026-11-27
- **核对范围：** 三份公开仓库的 README、讲义和入口；核对 Lecture 5 Notebook、TinyCal SDK、HTTP 服务及测试样例的衔接。未调用商业模型、启动 Temporal、运行 Harbor/Docker 或接入真实日历。
- **通读：** 已完成

| 官方入口 | 固定提交 | 包含与不包含 |
|---|---|---|
| [Lecture 5](https://github.com/cs329z/lecture5-frameworks) | [4e41828](https://github.com/cs329z/lecture5-frameworks/tree/4e418289b3032cb960984250395525eb91ae25c2) | `lecture5.ipynb`、TinyCal、合成邮件。Python 3.12；完整 Notebook 的模型段需 provider 凭据，TinyCal 仅需标准库。 |
| [HW1](https://github.com/cs329z/assignment1-harness) | [95f4cb7](https://github.com/cs329z/assignment1-harness/tree/95f4cb7bfe28a27a7cf5bcb1eccb1aaaf1385eb3) | 讲义、测试、类型与 adapter 骨架；核心题目接口故意未实现。确定性测试使用替身，live 测试另有真实模型成本。 |
| [HW2](https://github.com/cs329z/assignment2-evals) | [3a649d8](https://github.com/cs329z/assignment2-evals/tree/3a649d8da0dc6cb0422e6ce7ce7068c3b98be1db) | Harbor 示例任务、固定版本的被测框架、提交清单及写作模板；并非已完成的评测作业。 |

离线实测：令 `tinycal.server.make_server(0)` 只监听临时回环端口，给它临时 JSON 文件与虚构事件；通过真实 SDK 校验空日历、半小时事件、空档、修改、无效日期、文件写入及删除，共 7 项通过。未使用用户数据或课程账号。真实运行时序见 [课程节点](../../wiki/entities/stanford-cs329z-engineering-ai-agents.md)。
