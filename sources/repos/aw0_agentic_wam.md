# AW-0 官方代码

- **代码：** <https://github.com/ZhuoyangLiu2005/agentic-wam>
- **项目页：** <https://agentic-wam.github.io/>
- **论文：** <https://arxiv.org/abs/2610.30136>
- **权重：** <https://huggingface.co/miniFranka/agentic-wam-pretrain-yamego3k-soup-e1>
- **核对版本：** `275818e5787d59012eba4c43504342f1201ad25b`；核对日期 2026-11-01。
- **许可：** BSD 2-Clause；依赖模型各有许可。

仓库包含预训练、后训练、RoboTwin 和真机部署代码。`deploy/deploy_server.py` 的 `/act` 入口解析图像、14 维关节状态和指令，调用 `robotwin/deploy/deploy_policy.py` 的 `_infer_action_chunk`；启用代码专家时调用模型 `infer_action_trimodal`，将相对关节动作转换为绝对目标后返回。该服务返回字段 `code` 为 `None`，不可等同混合执行路径。

## 2026-11-21 复现资料核对

- **官方安装指南：** <https://github.com/ZhuoyangLiu2005/agentic-wam/blob/main/docs/INSTALL.md>，要求 Python 3.10、CUDA 12.8、torch 2.7.1+cu128；训练脚本默认多 GPU/H100 级配置。
- **预训练权重内容：** Hugging Face 模型卡列出 `agenticwam_pretrain.pt`、`config.yaml`、`dataset_stats.json`；后训练以 `paths.pretrain_ckpt` 加载。独立冻结组件需另行准备，不能将这个检查点误当成一键完成训练所需的全部权重。
- **程序标注：** <https://github.com/ZhuoyangLiu2005/agentic-wam/blob/main/docs/CODE_ANNOTATION.md>，训练脚本在 `scripts/posttrain_annotation_{yam,sharpa}.sh` 生成每帧程序标签，实际模型通过 `src/agenticwam/models/cosmos_mot/` 的联合推理接口读取。
- **后训练烟雾检查：** `DRY_RUN=1 bash scripts/train_real_yam.sh <dataset_dir> <code_cache.pt>` 只检查配置、数据和执行入口，**不会**运行训练或验证机器人效果。
- **额外公开权重：** 官方 README 还列出 RoboTwin 任务后训练 checkpoint：<https://huggingface.co/miniFranka/agentic-wam-posttrain-robotwin-suit>；不同于用户提供的 yam+ego 预训练权重。

预训练语料和预训练初始化所用 Cosmos-Predict2.5-2B checkpoint 未随仓库发布。参见[项目页归档](../sites/aw0_agentic_wam.md)与[论文归档](../papers/aw0_arxiv_2610_30136.md)。
