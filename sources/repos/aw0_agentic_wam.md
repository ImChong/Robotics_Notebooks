# AW-0 官方代码

- **代码：** <https://github.com/ZhuoyangLiu2005/agentic-wam>
- **项目页：** <https://agentic-wam.github.io/>
- **论文：** <https://arxiv.org/abs/2610.30136>
- **权重：** <https://huggingface.co/miniFranka/agentic-wam-pretrain-yamego3k-soup-e1>
- **核对版本：** `275818e5787d59012eba4c43504342f1201ad25b`；核对日期 2026-11-01。
- **许可：** BSD 2-Clause；依赖模型各有许可。

仓库包含预训练、后训练、RoboTwin 和真机部署代码。`deploy/deploy_server.py` 的 `/act` 入口解析图像、14 维关节状态和指令，调用 `robotwin/deploy/deploy_policy.py` 的 `_infer_action_chunk`；启用代码专家时调用模型 `infer_action_trimodal`，将相对关节动作转换为绝对目标后返回。该服务返回字段 `code` 为 `None`，不可等同混合执行路径。

预训练语料和预训练初始化所用 Cosmos-Predict2.5-2B checkpoint 未随仓库发布。参见[项目页归档](../sites/aw0_agentic_wam.md)与[论文归档](../papers/aw0_arxiv_2610_30136.md)。
