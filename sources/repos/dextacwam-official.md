# DexTacWAM — 官方实现与复现边界

- **仓库：** https://github.com/dextacwam/DexTacWAM
- **论文：** https://arxiv.org/abs/2609.24976v1
- **主页：** https://dextacwam.github.io/
- **权重/数据：** https://huggingface.co/collections/JensenYuan/dextacwam-6aba3aa25d3362d882cdefbc
- **核对范围：** 2026-11-13 阅读 README 中安装、数据、三阶段训练、离线评测、部署及许可；核对 main.py 训练/推理分发，runner/visual_vae_adapter_trainer.py 的 VisualVAEAdapterTrainer 和 main()、runner/tactile_dit_trainer.py 的 TactileDiTTrainer.prepare_dataset/prepare_models/train、runner/tactile_inferencer.py 的 TactileInferencer.infer，以及 deploy 分支 web_infer_scripts/run_server_tactile_relative.sh 的任务服务检查；未逐行审核全仓/运行 GPU 或真机。
- **通读：** 已完成所列官方核心入口与运行文档；不声称逐行审阅全部源码

## 真实代码入口

1. **预处理：** `scripts/get_statistics.py --relative --n-previous 4 --action-chunk 54` 生成任务统计，再运行 `scripts/preprocess_dex_vtam_cache.py --add-flow`。
2. **Stage 1：** `python -m runner.visual_vae_adapter_trainer --config configs/stage1_visual_vae_adapter_base.yaml`；再运行 `configs/stage1_visual_vae_adapter.yaml`。代码叫 visual_vae_adapter，本质是冻结 VAE 上的触觉编码器适配。
3. **Stage 2：** `torchrun --nproc_per_node=4 main.py --config_file configs/<task>/stage2_world_model.yaml --runner_class_path runner/tactile_dit_trainer.py --runner_class TactileDiTTrainer`。main.py 动态导入 runner，依次 `prepare_dataset → prepare_models → prepare_trainable_parameters → prepare_optimizer → prepare_for_training → prepare_trackers → train`。
4. **Stage 3：** 同一入口和 runner，但配置为 `stage3_action_expert.yaml`，需要第 2 阶段任务 checkpoint 做 warm-start，动作专家随机初始化。
5. **离线评测：** `main.py --mode infer --runner_class_path runner/tactile_inferencer.py --runner_class TactileInferencer --checkpoint_path ...`，调用 `prepare_val_dataset → prepare_models → infer`。
6. **真机（deploy 分支）：** `dryrun_relative_server.py` 检查张量布局 → `run_server_tactile_relative.sh` 校验模型/统计/布局并启服务 → `offline_client_relative.py` 离线回放 → `robot_client_tactile_sharpa_dexmate.py` 接收动作块（涉及真实机器人，应现场安全验收）。

## 开放、依赖与许可证

- **已公开：** 六任务配置、三阶段训练和离线推理源码、Stage 1 触觉编码器权重、488 条基础触觉示范（250 个指令，README 标约 3.5h，282 GB）与六任务各 96–100 条示范。
- **未公开：** Stage 2 世界模型、Stage 3 动作专家的作者任务 checkpoint。README 示例 Stage 2/3 参考配置使用 4× H200 NVL；需自训，不能直接加载作者真机结果模型。
- **混合许可：** DexTacWAM 原创及部分依赖代码 Apache-2.0；Genie-Envisioner 衍生文件 CC BY-NC-SA 4.0 非商业；上游 Genie / LTX 权重另受 LTX-Video Open Weights License 约束。不能将整个仓库标为可自由商业使用。
- **本次验证：** 对照入口，不涉及实际安装、性能复现或真机部署。

## 对 wiki 的映射

- [主实体](../../wiki/entities/paper-dextacwam.md)
- [项目归档](../sites/dextacwam-project.md)
- [论文归档](../papers/dextacwam_arxiv_2609_24976.md)