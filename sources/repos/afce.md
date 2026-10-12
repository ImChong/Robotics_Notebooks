# AFCE（AFCEProject/AFCE）

> 来源归档（repo；核对日期：2026-10-12）

- **代码：** <https://github.com/AFCEProject/AFCE>
- **论文：** <https://arxiv.org/abs/2610.13212>
- **项目页：** <https://afceproject.github.io/AFCE-Web/>
- **许可：** MIT（版权署名 Dexjoco contributors）
- **最近提交：** 2026-10-09（浅克隆 tip）
- **沉淀到 wiki：** [paper-afce-action-future-coupled-effect-vla.md](../../wiki/entities/paper-afce-action-future-coupled-effect-vla.md)

## 仓库内容（浅克隆核查）

- `afce/`：prepare / build_world / E 训练 / `export_effect_resume` / `export_joint_decoder` / `multihost_finger_delta_pi`（策略训练）/ `serve_finger_delta_pi`（推理服务）/ `finger_delta_eval_worker`（评测）。
- `effect_codec/`、`effect_vla/`：Effect 编解码与 DINOv3 证据辅助；`openpi/`：带 AFCE 钩子的 π0.5 训练/服务分支；`dexjoco/`：DexJoCo MuJoCo 环境与评测客户端；`configs/{rand_obj,rand_full,multi_task,ipad_reasoning}`；`experiments/finger_delta/`：规范实验协议与参考哈希；`scripts/pipeline/{prepare_statistics.py,train_e.sh,verify_source.py}`。
- 主流水线（README）：Effect 训练 → 联合动作解码器导出 → π0.5 单解码器 + GT-AUX + finger-Δ 策略 → DexJoCo 评测。
- **不含：** 数据集、DINOv3 权重、π0.5 基座参数（需 `action_dim=44`）、Effect 缓存、rollout 视频、真机遥操作栈。数据与基线参考 HF `DexJoCo/DexJoCo-Datasets-LeRobot` 与 `DexJoCo/DexJoCo-Pi05`。
- README 注明开源清理后部分辅助模块被删除，源码清单可能无法与私有实验快照逐位对齐。
- 本次未实际运行训练或评测（需 GH200 级算力与外部权重）。
