# GlanceWAM（linhanwang/GlanceWAM）

> 来源归档

- **标题：** GlanceWAM
- **类型：** repo
- **来源：** Virginia Tech / Drexel / Northeastern / Purdue
- **链接：** <https://github.com/linhanwang/GlanceWAM>
- **论文：** <https://arxiv.org/abs/2608.23927>
- **权重集合：** <https://huggingface.co/datasets/LinhanWang/GlanceWAM>
- **许可：** MIT
- **入库日期：** 2026-08-30
- **一句话说明：** 稀疏测试时想象 WAM：训练、LIBERO / RoboCasa 评测 sweep、HF 数据与检查点。
- **沉淀到 wiki：** [`wiki/entities/paper-glancewam.md`](../../wiki/entities/paper-glancewam.md)

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 自检前向 | `python glancewam/model/framework/wam/GlanceWAM.py` |
| 数据路径 | `python glancewam/dataloader/lerobot_datasets.py --config_yaml examples/Robocasa_kitchen/train_files/config_glancewam_robocasa_kitchen.yaml` |
| LIBERO 训练 | `bash examples/LIBERO/train_files/run_libero_glancewam.sh` |
| RoboCasa 训练 | `bash examples/Robocasa_kitchen/train_files/run_robocasa_kitchen_glancewam.sh` |
| LIBERO 评测 | `python tools/eval_libero_sweep.py --ckpt …/steps_15000_pytorch_model_ema.pt` |
| RoboCasa 评测 | `python tools/eval_robocasa_kitchen_sweep.py --ckpt … --gpus 0,1,2,3 --include-state` |
| 参考硬件 | 4×H200，全局 batch 128 |

## 开源边界（截至 2026-08-30）

- **已开源**：训练、评测 sweep、框架前向自检可辨识。
- **权重 / 数据**：HF `LinhanWang/GlanceWAM`（约 21 GB，含 LeRobot v3 与检查点）。
- **真机脚本：** `deployment/` 目录存在；论文主表为仿真。

## 固定版本源码复核（2026-11-13）

- 版本：[11ef12eacdb1dda1aca437134f5ccb221e280685](https://github.com/linhanwang/GlanceWAM/tree/11ef12eacdb1dda1aca437134f5ccb221e280685)。已读根README、LeRobot integration README、server_policy.py、goal_worker.py全文；GlanceWAM.py完整predict_action；modeling_glancewam.py的_submit_goal/_install_goal/_request_due/_goal_step_queued/_goal_step_chunk/predict_action_chunk完整函数。未运行模型或下载21GB bundle。
- 仿真framework的predict_action无goal_latent就同步_generate_goal_latent，有则复用；force_gen生成新值但继续用旧值，客户端模拟陈旧。不能把仿真sweep说作真实并行生成。
- LeRobot GoalWorker spawn子进程另载模型副本，以CPU tensor队列submit/poll/collect；gpu_busy事件让worker在Euler步间让GPU。首次同步调用可collect等待，RTC首chunk走无goal fallback、不等采样。goal_async默认false；没有「永不等待」的无条件保证。
- integration为部分、仅推理release，未含该路径训练代码和实际robot脚本；根仓库含仿真训练/评测入口与HF数据/权重链接。许可证按根MIT、integration文件Apache及骨干分别核对。
