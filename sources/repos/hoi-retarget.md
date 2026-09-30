# shinben0327/hoi-retarget

> 来源归档（repo）

- **名称：** hoi-retarget
- **类型：** repo / motion-retargeting / hoi / dataset-pipeline
- **URL：** <https://github.com/shinben0327/hoi-retarget>
- **论文：** [arXiv:2609.34674](../papers/hoi_retarget_arxiv_2609_34674.md)
- **项目页：** <https://shinben0327.github.io/hoi-retarget/> — [`sources/sites/hoi-retarget-shinben0327-github-io.md`](../sites/hoi-retarget-shinben0327-github-io.md)
- **机构：** ETH Robotic Systems Lab
- **许可证：** BSD-3-Clause（`hoi_retarget/gmr/` 为 MIT，fork 自 GMR）
- **入库日期：** 2026-09-30
- **一句话说明：** `hoi-retarget` CLI：IK+物体缩放 → 窗口 NLP 接触优化 → `kinematic_window.pkl` / `contact_window.pkl`；批量 `batch_retarget.sh`；Viser 接触编辑。

## 运行入口（README）

| 步骤 | 命令 / 模块 |
|------|-------------|
| 安装 | conda + `pip install -e .`；见 `docs/INSTALL.md` |
| 单 clip | `hoi-retarget --input_file ... --out_dir ...` |
| 文件夹 | `hoi-retarget --src_folder ... --tgt_folder ...` |
| 批量 | `hoi_retarget/tools/batch_retarget.sh clips.txt out_dir` |
| 模式 | `--mode contact`（默认）或 `kinematic`；`--robot unitree_h2`；`--object_scale` |

## 关键代码路径

| 路径 | 作用 |
|------|------|
| `hoi_retarget/cli.py` | `hoi-retarget` 入口 |
| `hoi_retarget/retargeting/` | Sec. III-A IK、物体缩放、物体系 contact target |
| `hoi_retarget/optimization/` | Sec. III-B 窗口 NLP（Pinocchio + CasADi/IPOPT） |
| `hoi_retarget/contact/` | Sec. III-C Viser 接触段编辑 |
| `hoi_retarget/datasets/` | InterMimic/OMOMO、CARI4D 读取 |
| `hoi_retarget/gmr/` | vendored GMR fork（IK 后端） |
| `assets/robots/{g1,h2}/` | Unitree 描述（BSD-3） |

## 开源边界（2026-09-30）

- **已开源：** 完整重定向管线、13 个 InterMimic 物体 mesh（MIT）、配置与渲染工具
- **外部：** SMPL-X body models、OMOMO/InterAct 源 motion（用户按 `DATA.md` 下载）
- **已发布数据：** HF `shinben0327/hoi-retarget`（重定向产物，非原始 mocap）
