# RoboUniverse-10K 官方评测实现

- **类型：** repo
- **链接：** https://github.com/Einsia/RoboUniverse-10K
- **论文：** https://arxiv.org/abs/2611.31002
- **项目主页：** https://lab.einsia.ai/robo-universe/
- **数据：** https://huggingface.co/datasets/Einsia/RoboUniverse-10K
- **许可证：** Apache-2.0（代码）
- **核对日期：** 2026-11-24
- **核对范围：** 官方 README 和完整 evaluate.py 静态阅读；未运行 GPU 仿真。

## 复现入口

- `evaluate.py list --data "$DATA" --split heldout500`：读取 catalog.csv 与 splits/heldout500.csv。
- `evaluate.py prepare`：materialize 校验 ZIP 和依赖哈希、解包、生成隔离凭据、配置 Docker Compose。
- `evaluate.py run`：显式接受 Isaac Sim EULA、指定新工作目录、case 或 split；subprocess 调用 Harbor，独立 verifier 给出结果；summarize.outcome 区分成功、失败和基础设施异常。
- README 固定 HF 数据 revision ae170e8093dd6061765c14f2fb6a209432517d85；模型服务使用 SGLang，评测端 Python 3.12、Docker/Isaac Sim 5.1。完整安装和模型配置见 README。

公开内容是评测实现与环境包，不应据此声称完整 Real2Sim builder 或训练脚本已公开。

- [论文归档](../papers/robouniverse_10k_arxiv_2611_31002.md)
- [项目页归档](../sites/robouniverse-10k.md)
- [知识实体](../../wiki/entities/paper-robouniverse-10k.md)
