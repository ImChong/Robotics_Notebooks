# galilai-group/prh-proof：PRH with JEPAs 官方代码

> 来源归档（repository · 官方开源实现）

- **类型：** 论文理论验证 / 训练与评测源码
- **仓库：** <https://github.com/galilai-group/prh-proof>
- **论文：** <https://arxiv.org/abs/2610.39302>
- **代码许可证：** MIT（仓库 LICENSE；数据及预训练模型分别遵循其许可）
- **核查日期：** 2026-11-11
- **核对范围：** README、docs/design.md、train.py、prh/views.py、prh/forward.py、evaluate.py、prh/align.py 与复现入口。
- **通读：** 已完成
- **关联实体：** [Proving and Improving the Platonic Representation Hypothesis with JEPAs](../../wiki/entities/paper-prh-jepas.md)

## 开源状态与构成

**已开源，存在可运行训练/评测入口。** 官方仓库提供论文复现实验和数值定理验证，并包含图像/视频描述生成代码，而非机器人运动策略训练或部署程序。

| 路径 | 对应工作 |
|---|---|
| `train.py` | 单一训练入口，按 `--modality image/text`、`--caption-views`、`--objective lejepa/ntp/decode` 切换实验臂 |
| `prh/views.py` | `TextView`、`_pick`、`view_transforms` 区分同一世界的多个观测与单次观测腐蚀 |
| `prh/forward.py` | `lejepa`、`next_token`、`decode` 的训练及验证 forward |
| `prh/theory.py` | 有限世界/核/谱理论的可执行检验 |
| `evaluate.py` | `embed`、`align`、`wit`、`pairfree`、`nuisance` 等评估子命令 |
| `prh/align.py` | `Rotation.fit` 使用 SVD/Procrustes；`unpaired` 用 Gromov-Wasserstein 与后续优化 |
| `caption.py`、`prh/captioner.py` | 对齐空间后评估/打包图像视频描述器 |
| `docs/reproduce.md`、`paper/runs.csv` | 论文表格和图形到运行参数、种子的复现索引 |

## 最小可复现路径

官方要求 Linux、Python ≥3.11、PyTorch ≥2.6；CPU 可运行部分单元测试，训练需要 GPU。具体依赖与下载条件以仓库 README 为准。

```bash
pip install -e ".[dev]"
pytest -m "not slow"
python -m prh.data
python evaluate.py embed dinov2 runs/ref/flickr_dinov2
python train.py --modality text --caption-views
python evaluate.py align runs/ref/flickr_dinov2 runs/runs/<text-run>
```

- 上述 `<text-run>` 是由训练命令生成的实际 run 路径占位符，不是可直接执行的字面目录。
- `train.py` 构建数据与多视图、网络和 `stable_pretraining.Module`，由 `spt.Manager(... )()` 调用训练/验证；`OnlineWriter` 存验证 embedding，`evaluate.py align` 在非测试图文对上拟合旋转并计算测试指标。
- 程序依赖 `stable-pretraining`；论文运行使用的 revision 由官方 `docs/design.md` / `docs/reproduce.md` 记录。Flickr30k、Recap-DataComp、ImageNet 等数据和部分 DINOv3 权重需要另行准备或授权。

## 使用边界

`--caption-views` 是 good-view 文本实验的关键开关；单个 caption 的 span masking 与 next-token prediction 属于不同对照。无法以离线跨模态对齐分数代替机器人闭环表现；复现论文规模训练还需要与其相当的数据量和多卡计算资源。
