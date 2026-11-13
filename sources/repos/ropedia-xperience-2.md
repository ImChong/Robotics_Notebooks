# Ropedia Xperience 2 官方源码：HOMIE→KOSMO 配准

> 来源归档（repo / runnable subset）

- **仓库：** <https://github.com/Ropedia/Ropedia-Xperience-2>
- **项目页：** <https://ropedia.com/xperience/2-research/>
- **论文：** <https://arxiv.org/abs/2611.17060>
- **数据：** <https://huggingface.co/datasets/ropedia-ai/Ropedia-Xperience-2>
- **许可：** Apache-2.0（仅仓库代码）；数据 CC BY-NC 4.0
- **核对日期：** 2026-11-13
- **核对范围：** README、docs/scene_registration.md 全文，pyproject.toml 入口，scene_registration/cli.py、sources.py、quality.py 和 align.py 的核心配准函数、examples/fit_synthetic_trajectory.py。
- **通读：** 已完成
- **开放状态：** 已开源可运行的单个 HOMIE 记录到 KOSMO 场景配准模块；不包含原始传感器建图、手/身体估计、全套 Real-to-Sim/重定向/遥操作/长时生成或整批数据处理软件。

## 运行入口（官方）

```bash
git clone https://github.com/Ropedia/Ropedia-Xperience-2.git
cd Ropedia-Xperience-2
pip install -e .
xperience-register --kosmo /path/to/scene --homie /path/to/recording --out results/take-01
```

- Python 3.10–3.12。扫描输入要求 `colmap/*.zip`（text 相机模型）、`images/*.zip` 和 `obj/*.obj`；录制输入要求 `mocap_main/f_slam.txt`、相机标定、立体视频；可信质量 PASS 需要深度及几何检验数据。仅有 3DGS 或点云无法执行该配准。
- `scene_registration/cli.py:run()` 依次调用 `Source`、`Homie`、`Kosmos`、`global_init`、`fine_localize`、`fit_and_eval`、`checks`、`quality.verdict`、`write_outputs`。
- 输出 `transform.json` 的 `S_kosmos_from_homie`，将 HOMIE SLAM 世界点映射至扫描场景坐标系；伴随 `metrics.json`、`pnp_frames.json`、变换后的设备/相机轨迹、可选手/身体 NPZ 及可视化。
- `PASS` 要求留出帧平移中位 ≤5 cm、p90 ≤12 cm，重力偏差 ≤2°、ICP 位移 ≤10 cm / 旋转 ≤2°，且独立检查完整；`VERIFIED` 仅验证视觉配对，不等于注册精度合格。
- `--no-depth` 无法获得 PASS；`examples/fit_synthetic_trajectory.py` 仅演示**变换拟合**，不验证图像匹配与深度。

## 对 wiki 的映射

- [Ropedia Xperience 2 实体页](../../wiki/entities/paper-ropedia-xperience-2.md)
- [论文归档](../papers/ropedia_xperience_2_arxiv_2611_17060.md)
