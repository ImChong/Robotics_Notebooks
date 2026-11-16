# SWAM 官方实现入口

- **仓库：** <https://github.com/shaaaaron/SWAM>
- **论文：** <https://arxiv.org/abs/2611.12870>
- **项目：** <https://char-lotte-xia.github.io/swam-official/>
- **数据：** <https://huggingface.co/datasets/FreDERicia/SWAM-Pretrain>
- **核验日期：** 2026-11-11
- **核对范围：** README、SETUP.md、examples/sft/train_dreamzero.sh、dreamzero/robotwin_eval/run_block_full50.sh、dreamzero/robotwin_eval/launch_eval_shard.sh；底层所有 Python 模块未全读。
- **通读：** 未完成

## 代码入口（已核对路径）

- 训练启动脚本 `examples/sft/train_dreamzero.sh` 先校验 RLINF_ROOT / DREAMZERO_PATH / GPU / batch / Ray 环境，rank0 启动 Ray head，再将训练交给 RLinf。
- RoboTwin 50 任务评测 `dreamzero/robotwin_eval/run_block_full50.sh` 按 GPU 选择固定任务分片和 clean/randomized 配置，调用 `launch_eval_shard.sh`。
- `launch_eval_shard.sh` 设置模型 checkpoint、RoboTwin 根目录、日志和输出目录，并调用 `robotwin_eval/start_eval.sh`。
- SETUP.md 指明模型服务为 `dreamzero/robotwin_eval/dreamzero_robotwin_server.py`、环境侧 `RoboTwin/script/eval_policy.py`。
- 数据、基础权重和仿真资产需单独准备；不同组件各有许可证。

## 对 wiki 的映射

- [SWAM 主节点](../../wiki/entities/paper-swam-cross-modal-alignment.md)
