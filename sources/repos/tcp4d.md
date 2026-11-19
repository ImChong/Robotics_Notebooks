# TCP4D 官方源码与运行入口

- **类型：** repo
- **代码：** https://github.com/Livioni/TCP4D
- **论文：** https://arxiv.org/abs/2611.23143
- **核对日期：** 2026-11-19
- **核对范围：** 官方 README、已公开的代码入口/脚本及许可说明；仅静态阅读，未执行训练、模型推理或真机实验。
- **通读：** 已完成
- **wiki：** [独立论文节点](../../wiki/entities/paper-tcp4d-2611-23143.md)
- **项目：** https://livioni.github.io/TCP4D/

## 开源与复现边界

Stage1 `infer_stage1.py` main→load_episode→initial_query→load_model→run_inference；Stage2 `infer_stage2.py` main→load_episode→load_stage2_policy→run_inference/_run_windows，32 步预测/20 步重规划。训练有 `train_stage1.py`、`train_stage2.py`，模型数据见 README/HF；未执行实际推理。

- 官方入口：python infer_stage1.py --checkpoint <path> --input <episode> --interactive；python infer_stage2.py --checkpoint <path> --input <episode> --interactive
