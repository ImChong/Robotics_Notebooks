# EasyMocap

> 来源归档

- **标题：** EasyMocap（Easy Human Motion Capture Toolbox）
- **类型：** repo
- **链接：** https://github.com/zju3dv/EasyMocap
- **文档站：** https://chingswy.github.io/easymocap-public-doc/
- **机构：** 浙江大学（ZJU）CAD&CG 三维视觉组（3DV）
- **许可证：** 自定义 **非商业科研许可**（教育 / 研究 / 非营利；商业使用须邮件 `xwzhou@zju.edu.cn`；衍生修改须开源且禁止商用）
- **星标 / 版本（入库日）：** ~4.8k stars；`setup.py` 标注 **v0.2.1**
- **入库日期：** 2026-09-04
- **最后更新：** 2026-09-04
- **一句话说明：** 浙大 3DV 的 **无标记人体动捕 + 稀疏多视角新视角合成** 工具箱：多相机拟合 SMPL / SMPL+H / SMPL-X / MANO，并接入互联网视频、镜面人、多人重建与 Neural Body。
- **沉淀到 wiki：** 是 → [`wiki/entities/easymocap.md`](../../wiki/entities/easymocap.md)

## 开源核查（步骤 2.5，2026-09-04）

| 项 | 结论 |
|----|------|
| 代码 | **已开源** — <https://github.com/zju3dv/EasyMocap>（可安装 CLI `emc`、多条 demo 脚本） |
| 文档站 | **已开源文档** — <https://chingswy.github.io/easymocap-public-doc/>（安装 / Quick Start / 各 setting） |
| 权重 / 人体模型 | **不随仓分发**：SMPL / SMPL-X / MANO / FLAME / VPoser 须自行向官方站点申请后放到 `data/bodymodels/` |
| 数据集 | **ZJU-MoCap（LightStage + Mirrored-Human）部分开放**：需签署 [协议 PDF](https://pengsida.net/project_page_assets/files/ZJU-MoCap_Agreement.pdf) 并邮件 Qing Shuai / cc Xiaowei Zhou 索取链接；另有公开 demo zip（Dropbox / 百度网盘） |
| Neural Body 训练 | **分仓** — <https://github.com/zju3dv/neuralbody>；本仓提供 `apps/neuralbody` 与 `requirements_neuralbody.txt` 入口 |
| iMocap 多视频特定动作 | README 标 **Coming soon**（论文 [arXiv:2008.07931](https://arxiv.org/abs/2008.07931) 已发表） |

## 仓库要点

### 快速入口

| 入口 | 说明 |
|------|------|
| `emc`（`apps/mocap/run.py`） | 配置驱动的通用 mocap 入口（`setup.py` `console_scripts`） |
| `apps/demo/mv1p.py` | 多视角单人 → SMPL / SMPL-X / MANO |
| `apps/demo/mvmp.py` | 多视角多人 3D 姿态（`config/exp/mvmp1f.yml`） |
| `apps/demo/auto_track.py` | 3D 关键点跟踪与插值，再接 SMPL 拟合 |
| `apps/demo/1v1p_mirror.py` / `mv1p_mirror.py` | 镜面人（Mirrored-Human） |
| `scripts/preprocess/extract_video.py` | 视频抽帧 + OpenPose / MediaPipe 等 2D 关键点 |
| `apps/calibration/` | 内参 / 外参标定 |
| `apps/annotation/` | bbox / 关键点 / mask 标注器 |
| `apps/vis3d/` | 实时 3D 可视化（骨架 / 网格） |

### 典型数据布局

```text
<seq>/
├── intri.yml
├── extri.yml
└── videos/*.mp4
```

输出默认 JSON：`keypoints3d/*.json`（OpenPose Body25 + 置信度）与 `smpl/*.json`（`Rh` / `Th` / `poses` / `shapes`；**`poses` 前 3 维置零，全局朝向单独放在 `Rh`**，与官方 SMPL `global_orient` 不等价）。可再导出 BVH（需 SMPL-maya FBX）。

### 依赖与安装（文档站）

- 官方一键示例：`conda` Python **3.9** + CUDA 11.1 / 11.6 + PyTorch 1.9.1 / 1.12.0 + `python setup.py develop`。
- Neural Body：`requirements_neuralbody.txt` + `spconv`；AniNerf 另装 PyTorch3D。
- 人体先验可选：`scripts/install/install_vposer.sh`。
- Colab demo：README 链到 Google Colab notebook。

### 配套论文（工具箱声明的工作子集）

| 工作 | 会议 / 出处 | 在本仓中的角色 |
|------|-------------|----------------|
| iMocap | ECCV 2020，[arXiv:2008.07931](https://arxiv.org/abs/2008.07931) | 互联网多视频特定动作（代码 Coming soon） |
| Mirrored-Human | CVPR 2021，[arXiv:2104.00340](https://arxiv.org/abs/2104.00340) | 镜面视频拟合；分仓 `zju3dv/Mirrored-Human` |
| mvpose | T-PAMI 2021 | 多视角多人；`apps/demo/mvmp.py` |
| Neural Body | CVPR 2021，[arXiv:2012.15838](https://arxiv.org/abs/2012.15838) | 稀疏视角动态人新视角合成 |
| MultiNeuralBody | SIGGRAPH 2022 | 人–人交互新视角合成（2022-11 合入） |

## 对 wiki 的映射

- 文档站：[easymocap-public-doc.md](../sites/easymocap-public-doc.md)
- 实体页：[wiki/entities/easymocap.md](../../wiki/entities/easymocap.md)
- 同组后续单目 HMR：[gvhmr.md](gvhmr.md) / [wiki/entities/gvhmr.md](../../wiki/entities/gvhmr.md)
