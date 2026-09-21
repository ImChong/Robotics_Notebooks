# weiyufei0217.github.io/G2G（G2G 项目页）

> 来源归档（ingest）

- **标题：** G2G: Exploiting Intra-Group Geometry for Inter-Group Pose Estimation
- **类型：** site / project-page
- **官方入口：** <https://weiyufei0217.github.io/G2G/>
- **入库日期：** 2026-09-21
- **一句话说明：** 浙江大学等团队的 G2G 论文配套站：框架图、跨序列重定位与多相机 rig 里程计 demo、四数据集定量表、FoV overlap 分析与交互重建 viewer。

## 页面公开信息（检索自 2026-09-21）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://weiyufei0217.github.io/G2G/> |
| arXiv | <https://arxiv.org/abs/2606.08284> |
| 代码 | <https://github.com/WeiYuFei0217/G2G> |

## 源码开放核查（步骤 2.5）

- **开放程度：已开源（代码 + 权重下载指引）**
  - **项目页 Footer / BibTeX 区** 写明 *Code is available at github.com/WeiYuFei0217/G2G*。
  - **仓库（2026-09-21 核查）：** 含 `scripts/train_reloc.py`、`train_rig.py`、`eval_reloc.py`、`eval_rig.py`；vendored `third_party/mapanything/`；权重托管 Baidu / Google Drive / Hugging Face。
  - **预训练：** 10 组 task×dataset 权重 + MapAnything large backbone checkpoint（外站，README 逐步说明）。
  - **许可：** CC BY-NC 4.0。

## 与论文一致的公开主张（便于 wiki 溯源）

1. **Framework：** 冻结 MapAnything（DINO-V2 ViT-L/14）编码每组几何；Perceiver resampler + cross-group merged self-attention bridge + multi-frame pose head 共 ~32M 可训练参数。
2. **Task 1 — Cross-Sequence Relocalization：** 不同时间两短序列 → 相对对齐位姿；Table 1 四数据集 mean errors。
3. **Task 2 — Multi-Camera Rig Odometry：** 相邻时刻两 rig → inter-rig motion；Table 2 六配置（HM3D 8/4-cam、TartanGround、NCLT、ZJH sim/real）。
4. **Overlap 分析：** NCLT median FoV overlap **0.24** 最难；G2G 低 overlap 区间 rotation/translation error 退化最缓。
5. **Qualitative：** ZJH sim-to-real 墙两侧弱重叠仍对齐；NCLT 跨季节大外观变化 residual rot **1.12°** / trans **5.3 cm**。

## 对 wiki 的映射

- [`wiki/entities/paper-g2g.md`](../../wiki/entities/paper-g2g.md) — 方法栈、开源状态与任务定位
- [`sources/repos/g2g.md`](../repos/g2g.md) — 仓库与复现入口归档
- [`sources/papers/g2g_arxiv_2606_08284.md`](../papers/g2g_arxiv_2606_08284.md) — 论文级摘录
