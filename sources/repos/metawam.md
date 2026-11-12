# showlab/MetaWAM — 官方代码与复现入口

- **仓库：** https://github.com/showlab/MetaWAM
- **论文：** https://arxiv.org/abs/2611.14365
- **权重：** https://huggingface.co/Ethanliang99/MetaWAM-RoboCasa-Predict2-2B
- **核对范围：** README、ROBOCASA.md、scripts/train_robocasa.sh、scripts/test_robocasa.sh 与 cosmos_policy/models/policy_text2world_model.py 频域槽位实现；其他源码与测试仍待通读。
- **通读：** 未完成

## 真实代码入口

- **RoboCasa 训练：** scripts/train_robocasa.sh → torchrun -m cosmos_policy.scripts.train，experiment=metawam_robocasa。
- **RoboCasa 评测：** scripts/test_robocasa.sh → python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval。
- **元帧编码：** cosmos_policy/models/policy_text2world_model.py 包含 task/current proprio/action/future proprio/value 的频域通道槽位与 IDCT 解码。
- **已发布权重：** RoboCasa checkpoint 与数据统计及 T5 embeddings；下载路径与环境依赖依官方 README。

## 复现约束

评测脚本默认 24 tasks × 50 trials × seeds 195、196、197；训练默认 4 GPU。未在本环境执行训练、评测或验证论文指标。

## 对 wiki 的映射

- [MetaWAM](../../wiki/entities/paper-metawam.md)
