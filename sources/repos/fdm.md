# FDM 官方仓库

> 来源归档

- **标题：** LogosRoboticsGroup/FDM
- **类型：** repo
- **代码：** <https://github.com/LogosRoboticsGroup/FDM>
- **论文：** <https://arxiv.org/abs/2610.13117>（[论文归档](../papers/fdm_arxiv_2610_13117.md)）
- **项目页：** <https://logosroboticsgroup.github.io/FDM/>（[站点归档](../sites/fdm-logosroboticsgroup-github-io.md)）
- **许可：** MIT
- **入库/核查日期：** 2026-10-12（浅克隆；tip 提交 2026-10-09 “开源 Forward Dynamics Model 代码与项目主页”；约 400 个文件）
- **沉淀到 wiki：** [`wiki/entities/paper-fdm-forward-dynamics-models.md`](../../wiki/entities/paper-fdm-forward-dynamics-models.md)

## 仓库入口

实现构建在 StarVLA 之上并保留 `starVLA` 包名。

| 路径 | 作用 |
|------|------|
| `starVLA/model/framework/VLM4A/Pi05_causal.py` | FDM w/ VLM（`forward`、`predict_action`、可选 `predict_video`） |
| `starVLA/model/framework/WM4A/WanMoT_causal.py` | FDM w/ VGM |
| `starVLA/model/modules/pi05/causal.py`、`starVLA/model/modules/wan_mot/` | 因果专家与 MoT |
| `examples/LIBERO/train_files/train_pi05-causal.sh`、`train_wanmot-causal.sh` | LIBERO 训练启动 |
| `examples/LIBERO/eval_files/run_policy_server.sh`、`eval_libero.sh` | 策略服务与评测 |
| `examples/Robotwin/`、`examples/ARX/` | RoboTwin / ARX 真机的数据注册与训练脚本 |
| `deployment/` | 策略服务、机器人推理、遥操作 |
| `scripts/vla/precompute_text_embeds.py` | VGM 文本嵌入缓存 |

## 边界

- 环境：Python 3.10、PyTorch 2.7.0、CUDA 12.8，需 flash-attn、特定 LeRobot commit 与 decord。
- 数据集、FDM checkpoint、文本缓存与日志不随仓库发布；README 注明部分默认值与论文不同，需按复现说明覆盖。
