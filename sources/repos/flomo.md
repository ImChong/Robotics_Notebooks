# FloMo 官方仓库

> 来源归档（repo）；2026-11-19 核对 README 与可运行代码入口

- **标题：** FloMo: 3D Scene Flow as a World Action Model Intermediate for Learning from Human Video
- **类型：** repo
- **链接：** <https://github.com/pairlab/flomo>
- **论文：** <https://arxiv.org/html/2611.05062>
- **项目页：** <https://flomo-wam.github.io/>
- **权重：** <https://huggingface.co/shitijgovil/FloMo>
- **许可：** MIT（GitHub 仓库 LICENSE）
- **入库日期：** 2026-11-05；源码核对：2026-11-19
- **核对范围：** README 安装、权重、推理、RoboTwin 微调、自定义训练、预处理与模块布局；`pyproject.toml`、`train.py` 主训练逻辑、`infer_example.py`、`flomo/models/FloMo.py` 的 `forward`/`infer`、`flomo/loaders/cotrain_loader.py`、`preprocessing/preprocess_robotwin.py` 入口。
- **通读：** 已完成
- **沉淀到 wiki：** [paper-flomo](../../wiki/entities/paper-flomo.md)

## 开源状态与真实边界

**已公开可运行代码及权重：** `train.py`、`infer_example.py`、`preprocessing/`、`cfg/`；HF 提供 MA（motion+action）、VA（video+action）、VMA（video+motion+action）三个 checkpoint。仓库 README 说明需自行安装 Wan2.2、DreamZero 依赖并下载 Wan VAE/umT5。**未在本次环境安装依赖或实际运行模型**，不能以源码阅读等价于复现成功。

## 官方可运行入口

| 入口 | 代码职责 | 必要条件 |
|------|----------|----------|
| `python -m preprocessing.preprocess_robotwin mode=tracks ...` | 以 SpatialTrackerV2 提取 RoboTwin 三维点轨迹 | 官方 demos、相应 tracker 与 cfg |
| `python -m preprocessing.preprocess_robotwin mode=text_emb ...` | 缓存指令的 umT5 嵌入 | Wan 文本编码器 |
| `python train.py --config-name train_robotwin ... resume=false` | 从官方 checkpoint 权重微调、加载数据与损失 mask、保存 checkpoint | 50 条 RoboTwin 示范与预处理产物 |
| `torchrun --nproc_per_node=8 train.py datasets=flomo ...` | 对官方 EgoVerse / MolmoAct2 / RealYAM 混合数据共训练 | 合法数据路径、多 GPU 环境 |
| `python infer_example.py --checkpoint ... --image ... --task ... --ee-pose=...` | 加载模型、构造图像/文本条件、调用 `FloMo.infer`、反归一化双臂动作 | CUDA、模型权重、三视角图像、当前末端位姿 |

## 真实调用关系核对

- `train.py main()` → `build_cotrain_dataset` / `build_cotrain_dataloader` → `FloMo(cfg).setup_model()` → `model(batch)` → `FloMo.forward()` 与 backbone；`accelerator.backward(loss)`、`optimizer.step()`、`save_training_checkpoint()`。
- `FloMo.forward` 从模态 mask 读取输入/目标、共享 DiT 前向、按输出目标计算 per-sample loss，再通过 `cotrain_loss` 汇总；无 action label 的视频按 mask 不产生动作损失。
- `infer_example.py` 经 `load_model`、`build_batch` 调用 `model.infer(batch, args.inference_steps)`；转换动作到 14D 双臂末端格式并打印，不直接控制真实硬件。
- 论文动作训练空间采用 20D（双臂 3D 位置、6D 旋转、1D 夹爪），而 CLI 报告的 `(T,14)` 是推理后处理输出，不能混同。

## 依赖提示

官方 README 指定 Python 3.10、CUDA GPU；其安装示例是 PyTorch 2.10/cu128，SpaTrackerV2 需要 `torchvision<=0.25` 和 `numpy<2`，提醒不要直接安装 SpaTrackerV2 自带 requirements。完整安装命令及可能变更依 [官方 README](https://github.com/pairlab/flomo#install) 为准。
