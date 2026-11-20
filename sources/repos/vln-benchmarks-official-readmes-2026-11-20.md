# VLN 七类基准与工具链：官方 README 和评测入口核对（2026-11-20）

- **类型：** repo / dataset / evaluation / primary-source reading
- **入库日期：** 2026-11-20
- **核对范围：** 已通读 R4R（google-research/r4r/README.md）、CVDN（mmurray/cvdn/README.md）、HANNA（khanhptnk/hanna/README.md）及 SOON（ZhuFengdaaa/SOON/README.md）的官方 README；另核对 Matterport3D Simulator 的 R2R/数据许可部分、VLN-CE 的任务/环境安装部分、REVERIE 的任务定义/版本说明与 NaviLLM 的基准支持/安装部分。未据此声称通读七篇论文全文；R2R 与 SOON 的论文原文阅读已在仓库各自 source 单独存档。
- **通读：** 已完成

## 官方资料映射（非论文全文转载）

| 任务 | 原论文 | 官方数据/实现 | README 核对结论 |
|---|---|---|---|
| R2R | [Anderson et al., CVPR 2018](https://arxiv.org/abs/1711.07280) | [Matterport3DSimulator / tasks/R2R](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R) | 90 个 Matterport3D 场景，离散全景导航图；R2R 提供数据下载、train.py、eval.py、EvalAI |
| VLN-CE | [Krantz et al., ECCV 2020](https://arxiv.org/abs/2004.02857) | [jacobkrantz/VLN-CE](https://github.com/jacobkrantz/VLN-CE) | Habitat 低层连续空间动作评测；官方仓同时支持 R2R/RxR，旧版本依赖 Habitat 0.1.7 |
| R4R | [Jain et al., ACL 2019](https://aclanthology.org/P19-1181/) | [google-research/r4r](https://github.com/google-research/google-research/tree/master/r4r) | 不重新发布 R2R 原数据；用 `r4r_generate_data.py` 拼接路径，基于距离阈值 3.0 m，可选航向阈值 |
| CVDN / NDH | [Thomason et al., 2019](https://arxiv.org/abs/1907.04957) | [mmurray/cvdn](https://github.com/mmurray/cvdn) | 2,050 段对话、7k+ 轨迹、83 栋房屋；`tasks/CVDN/data/download.sh` 与 `tasks/NDH/train.py`、`eval.py` |
| HANNA | [Nguyen & Daumé III, EMNLP 2019](https://aclanthology.org/D19-1063/) | [khanhptnk/hanna](https://github.com/khanhptnk/hanna) | 主仓 README 给交互式视觉/语言求助任务说明；数据及实验入口指向 `hanna-private` 路径，需单独核验其访问性 |
| SOON / FAO | [Zhu et al., CVPR 2021](https://arxiv.org/abs/2103.17138) | [ZhuFengdaaa/SOON](https://github.com/ZhuFengdaaa/SOON) | `r2r_src/train.py --dataset SOON --det_loss` 训练、`run/eval.sh` 评测；Matterport3D 与早期 Python 3.6 依赖 |
| REVERIE | [Qi et al., CVPR 2020](https://arxiv.org/abs/1904.10151) | [YuankaiQi/REVERIE](https://github.com/YuankaiQi/REVERIE) | 导航 + 远程实例指认；官方说明原版与 v2 数据/grounding 方案并存，2026-01 榜单搬迁、2026-03 盲测手动收件 |

## 框架原始入口

- [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator)：离散视点图和 R2R 基准。场景需依 Matterport3D 授权单独下载。
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)：连续具身任务与实验；[Habitat-Sim](https://github.com/facebookresearch/habitat-sim) 是仿真后端。
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE)：Habitat 上 R2R-CE / RxR-CE 任务配置与基线。
- [NaviLLM](https://github.com/zd11024/NaviLLM)：作者 README 明列支持 CVDN、SOON、R2R、REVERIE 与 ScanQA，属于模型/统一任务框架而非基准定义者。
- [DUET](https://github.com/cshizhe/VLN-DUET)：作者源码可用于 R2R / SOON / REVERIE 的离散导航对照。

## 官方代码运行入口核对（仅记录实际可见入口）

```bash
# R4R：输入原始 R2R 路径与 connectivity，再生成长轨迹
python r4r_generate_data.py --input_file_path=R2R_train.json --output_file_path=R4R_train.json --connections_dir=connectivity --distance_threshold=3.0

# CVDN / NDH：原仓 README 指定的基线入口
python tasks/NDH/train.py --path_type=trusted_path --history=all --feedback=sample --eval_type=val
python tasks/NDH/eval.py

# SOON：原仓 README 指定的基线入口
python3.6 -u r2r_src/train.py --dataset SOON --maxAction 20 --log_every 500 --rl_ml_weight 1 --det_loss
sh run/eval.sh
```

## 度量与协议资料

- R2R：官方 [R2R 任务 README](https://github.com/peteanderson80/Matterport3DSimulator/blob/master/tasks/R2R/README.md)，导航误差与成功率以作者论文与官方脚本为准。
- R4R：官方 [README](https://github.com/google-research/google-research/blob/master/r4r/README.md) 与 [ACL 论文](https://aclanthology.org/P19-1181/) 提出非最短路路径跟随和 CLS。
- CVDN：官方 [NDH eval.py](https://github.com/mmurray/cvdn/blob/master/tasks/NDH/eval.py) 与 [EvalAI](https://eval.ai/web/challenges/challenge-page/463/overview)，区分 Goal Progress 与终点导航指标。
- VLN-CE：官方 [仓库与 challenge](https://github.com/jacobkrantz/VLN-CE)，区别离散图路径与连续运动。
- SOON：官方 [SOON 代码](https://github.com/ZhuFengdaaa/SOON) 与 [论文](https://arxiv.org/abs/2103.17138)，判定需同时考虑停点与目标朝向。
- REVERIE：官方 [仓库与最新评测通知](https://github.com/YuankaiQi/REVERIE)，区分导航成功和目标实例定位成功；不同数据版本勿合并统计。
- nDTW / SDTW：[Ilharco et al., General Evaluation for Instruction Conditioned Navigation Using Dynamic Time Warping](https://arxiv.org/abs/1907.05446)：是后续轨迹一致性指标，不应误写成 R2R 2018 首发的原始指标。

## 复现边界

1. 所有 benchmark 必须保存训练/评测 split、场景、路径长度、动作空间、成功半径、STOP 协议与模型输入；不能只比某一个 SR。
2. 数据集注释和 Matterport3D 扫描不适用同一授权：即使代码 MIT，也不代表扫描图像可以随意重新分发。
3. 此归档为各作者官方 README 的技术与入口核对摘要，不包含论文全文、测试集或原始图像数据；对论文中的每个数量及消融，应回到对应论文原文与既有论文节点进一步核查。

## 对 wiki 的映射

- [VLN 七基准、框架与评测对照](../../wiki/comparisons/vln-seven-benchmarks-frameworks-evaluation.md)
- [VLN 任务总览](../../wiki/tasks/vision-language-navigation.md)
