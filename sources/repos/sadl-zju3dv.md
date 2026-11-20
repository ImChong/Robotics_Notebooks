# zju3dv/SADL 官方代码归档

- **类型：** repo；该代码只作为论文的配套资料，不新建第二个项目实体。
- **代码：** https://github.com/zju3dv/SADL
- **论文：** https://arxiv.org/abs/2611.21596
- **项目：** https://zju3dv.github.io/SADL/
- **模型和数据：** https://huggingface.co/DavidSunok/SADL
- **开放状态：** 已公开训练和评测入口，README 提供 Hugging Face 权重及 HDF5 数据下载命令。
- **核对范围：** 已静态阅读 README、docs/TRAINING.md、train.py、evaluate.py、src/train_mixed.py、src/engine.py、src/model/collapse_study/regularizers.py 的主训练与损失关键路径；其他源码及内核未全部阅读，没有实际运行。
- **通读：** 未完成

## 核对的具体执行入口

| 文件 | 职责 |
|---|---|
| train.py | 根据 --cell、--gpus、--datasets 启动 torch.distributed.run，4 GPU 运行 src/train_mixed.py |
| src/train_mixed.py | 构建任务数据集、共享模型及优化器，调用 backward_step，保存 runs/<cell>/last.ckpt 和导出权重 |
| src/engine.py / MixedModel.visual | ViT 编码器取 CLS 表征 H，经 BN projector 得 Z |
| src/engine.py / MixedModel.task_loss | 潜在转移预测，INTACT 还计算意图动作损失，LeWM 分支禁用该损失 |
| src/engine.py / MixedModel.regularization | 对齐并汇总多卡 H、Z 样本，调用正则器 |
| src/model/collapse_study/regularizers.py | CanonicalCapacityMatching：计算 SIGReg、容量代理、stop-gradient、单侧 SADL 损失 |
| src/engine.py / backward_step | 汇总任务与正则损失，反向传播、诊断和梯度归约 |
| evaluate.py | 根据 checkpoint 和 planner 运行 eval/protocol.py，INTACT 用 direct / guarded_a，LeWM 用 cem |

## 官方 README 复现命令（本次未执行）

```bash
python -m pip install -r requirements.txt
hf download DavidSunok/SADL --include "datasets/*.h5" --local-dir hf
python tools/prepare_datasets.py --root hf/datasets --output datasets.json
python train.py --cell INTACT_4T_S3072 --gpus 0,1,2,3 --datasets datasets.json
python evaluate.py --checkpoint runs/INTACT_4T_S3072/exports/reacher --task reacher --data-dir hf/datasets --planner direct
```

官方训练指南：单机 4 GPU / global batch 256 个窗口；SIGReg 0.015、SADL 0.01 为所列示例配置，论文各表的正则权重可能不同。评测默认 seeds 0/1/42 各 100 episodes。

## 局限

公开源码并不意味着在本环境测试成功；还需数据依赖、GPU、分布式 BN、数值精度与 planner 协议的实际验证。

## 关联

- [论文来源](../papers/sadl-arxiv-2611-21596.md)
- [项目主页](../sites/sadl-project-page.md)
- [论文详情与时序图](../../wiki/entities/paper-sadl-2611-21596.md)
