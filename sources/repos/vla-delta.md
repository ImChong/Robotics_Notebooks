# VLA-Δ 官方代码 / Source Repository

- **仓库：** <https://github.com/ZJU-OmniAI/VLA-Delta>
- **项目：** <https://zju-omniai.github.io/VLA-Delta/>
- **论文：** <https://arxiv.org/abs/2610.21306>
- **许可证：** Apache-2.0（主仓；THIRD_PARTY_NOTICES 另列第三方约束）
- **核验日期：** 2026-11-16
- **核对范围：** 完整 README；静态审阅 scripts/train.py、scripts/evaluate.py、scripts/prepare_pretraining.py、server.py、utils/websocket_server_policy.py、evaluation/libero/run_eval.py、configs/finetune/libero.yaml。
- **通读：** 已完成
- **开放程度：** 已公开三阶段训练、仿真评测、微调代码及三个模型检查点；完整 484h 标注资源公开入口未确认。此次未运行 GPU 训练与仿真。

## 官方运行入口与实际模块关系

1. **预训练数据准备：** scripts/prepare_pretraining.py --config configs/pretrain/prepare.yaml → utils.pretraining_preparation 的 entries、sidecars、sources、slots、freeze。代码注释指出基于已有 Stage05/Slot 标注制备训练产物，非从头调用全部 Qwen/SAM/MegaFlow 标注服务。
2. **三阶段预训练：** scripts/train.py 分别加载 configs/pretrain/stage1_ar.yaml → stage2_aux.yaml → stage3_joint.yaml；通过 utils.cli_options.parse_train_options 和 utils.portable_run.configure_run 建立配置，subprocess 调用 accelerate.commands.launch 启动 train_vla.py。
3. **微调：** scripts/train.py --config configs/finetune/libero.yaml --execute（或 robotwin.yaml）；必须显式 --execute，缺省打印计划命令；可用 --dry-run 预检。
4. **模型服务：** server.py --config configs/eval/libero_server.yaml → configure_datasets → policies.reasoning_vla_policy.ZR0Policy → WebsocketPolicyServer.serve_forever；收到 obs 时调用 policy.infer(obs) 传回 action chunk。
5. **仿真评测：** scripts/evaluate.py --config configs/eval/libero.yaml --execute 组织 evaluation.libero.run_eval；客户端读取 LIBERO 环境图像、状态、task，经 WebSocket 发往模型服务，取得 actions 后分步调用环境 step，输出 summary.json 和录像。RoboTwin 支路调用 evaluation/robotwin/run_eval.py。
6. **依赖：** Linux + Python 3.10 + CUDA，README 建议 torch 2.6/cu124；仿真依赖在独立环境安装，数据路径/权重/配置要由使用者准备。

## 对 wiki 的映射

- [论文实体与真实源码运行时序图](../../wiki/entities/paper-vla-delta.md)
- [原论文摘录](../papers/vla_delta_arxiv_2610_21306.md)
- [官网核查](../sites/vla-delta.md)
