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
