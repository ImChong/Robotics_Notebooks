# G2G（WeiYuFei0217/G2G 官方仓库）

- **标题**: G2G: Exploiting Intra-Group Geometry for Inter-Group Pose Estimation
- **论文**: <https://arxiv.org/abs/2606.08284>
- **项目页**: <https://weiyufei0217.github.io/G2G/>
- **代码**: <https://github.com/WeiYuFei0217/G2G>
- **类型**: code-release（与 `sources/papers/g2g_arxiv_2606_08284.md` 分工：本文件聚焦仓库、权重与训练/评测入口）
- **机构**: 浙江大学（ZJU）工业控制技术国家重点实验室等
- **License**: **CC BY-NC 4.0**
- **首次入库**: 2026-09-21

## 一句话摘要

官方实现：在 **冻结 MapAnything large/1024** 上仅训练 **32M** 跨组模块，用 **relative pose** 监督统一 **跨序列重定位** 与 **多相机 rig 里程计**；提供 HM3D/TartanGround/NCLT/ZJH 十组权重、数据预处理 pipeline 与 4-GPU 训练脚本。

## 仓库结构（README 摘要，截至 2026-09-21）

| 路径 | 作用 |
|------|------|
| `third_party/mapanything/` | **vendored MapAnything v1.0.1**（large/1024）；含 `return_info_sharing_features=True` 补丁 |
| `scripts/train_reloc.py` | Task 1 训练（支持 `--curriculum`） |
| `scripts/train_rig.py` | Task 2 训练 |
| `scripts/eval_reloc.py` / `eval_rig.py` | 单/多 GPU 评测 |
| `scripts/extract_g2g_weights.py` | 从含冻结骨干的 checkpoint 提取 G2G-only 权重 |
| `configs/reloc/*.yaml` / `configs/rig/*.yaml` | 数据集与路径配置（占位 `/path/to/...` 需替换） |
| `data_preprocessing/` | HM3D 等六步预处理（overlap matrix、window index） |
| `release_weights/` | 10 组 G2G 权重落盘目录（外站下载） |
| `map-anything-model/` | 冻结骨干 checkpoint（~2.1 GB，外站下载） |
| `examples/` | sanity-check 样例 bundle |

## 权重与下载

| 资产 | 入口 |
|------|------|
| G2G 十组权重 + 骨干 + 样例 + eval CSV | [Baidu Cloud](https://pan.baidu.com/s/17Z3jKvIYj_miHSiaQ_8Ctg?pwd=8888)（8888）/ [Google Drive](https://drive.google.com/drive/folders/1z6RfJT5i8n5C9YaZSwGyzv9LdbEQWcq5?usp=sharing) / [HF feixue22/G2G](https://huggingface.co/feixue22/G2G) |
| MapAnything backbone | 必须用 **large/1024**（v1.0.1）；**勿**用 2025-12 后 HF giant 权重 |

| Weight 文件 | 任务 | 数据集 |
|-------------|------|--------|
| `HM3D-Reloc.pth` | Reloc | HM3D |
| `TartanGround-Reloc.pth` | Reloc | TartanGround |
| `NCLT-Reloc.pth` | Reloc | NCLT |
| `ZJH-Reloc.pth` | Reloc | ZJH |
| `HM3D-Rig-8.pth` / `HM3D-Rig-4.pth` | Rig | HM3D |
| `TartanGround-Rig-4.pth` | Rig | TartanGround |
| `NCLT-Rig-Intra.pth` / `NCLT-Rig-Cross.pth` | Rig | NCLT |
| `ZJH-Rig-4.pth` | Rig | ZJH |

## 可复现入口（README）

```bash
# 环境
conda create -n g2g python=3.12 -y && conda activate g2g
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ./third_party/mapanything
pip install -e .
pip install -r requirements.txt

# Reloc 训练
torchrun --nproc_per_node=4 scripts/train_reloc.py --config configs/reloc/hm3d.yaml --curriculum

# Reloc 评测
python scripts/eval_reloc.py --config configs/reloc/hm3d.yaml \
  --checkpoint release_weights/HM3D-Reloc.pth --output-dir outputs/eval_HM3D-Reloc \
  --batch-size 16 --min-overlap 0.1
```

## 与机器人 / 多视图几何栈的关系

- **上游：** 每组需 **已知 intra-group extrinsics**（VO、标定或预建地图）；不是从单目 RGB 盲估全栈 SLAM。
- **下游：** 跨序列地图对齐、多相机 rig 轨迹拼接、多会话 relocalization；可与 [State Estimation](../../wiki/concepts/state-estimation.md) 链路上的 Glob3R / UniSim-SLAM 等 **互补**（G2G 专注 **组间相对位姿** 而非稠密建图全栈）。
- **骨干生态：** 依赖 [MapAnything](https://github.com/facebookresearch/map-anything) + DINOv2；与 VGGT 系方法（Reloc3R 等）在同一 **foundation model + 轻量头** 设计空间。

## 对 Wiki 的映射

- **`wiki/entities/paper-g2g.md`**：论文实体与方法归纳。
- **`sources/papers/g2g_arxiv_2606_08284.md`**：论文级摘录。
- **`sources/sites/g2g-weiyufei0217-github-io.md`**：项目页核查归档。
