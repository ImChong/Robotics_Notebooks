# TAMU-Visual-AI/4D-HOF 官方实现归档

- **GitHub：** https://github.com/TAMU-Visual-AI/4D-HOF
- **论文：** https://arxiv.org/abs/2610.08782
- **项目：** https://tamu-visual-ai.github.io/4D-HOF/
- **模型与预处理：** https://huggingface.co/eggplant71/4D-HOF
- **核验日期：** 2026-11-12
- **核对范围：** README、infer_ho3d.py、eval_ho3d.py、hof/inference/runner.py，以及 hof/inference/ 与 hof/model/ 目录结构。
- **通读：** 已完成

## 目前开放程度

- **部分开源（不是零起点端到端复现）**：仓库 README 明言 `The current release includes inference code for HO3D`，存在 `infer_ho3d.py`、`eval_ho3d.py`、`hof/` 推理模型代码及检查点下载。
- 安装需 NVIDIA GPU、CUDA、Conda、C++ 等；`bash setup_env.sh`；MANO_RIGHT.pkl 须用户另行按 MANO 授权下载，HO3D_v3 数据集也需另取。
- `bash download_assets.sh` 下载 `checkpoints/4dhof.pth` 和 `_DATA/ho3dv3_init/<sequence>/`，包含 hand.pkl / mesh.pkl / pose.pkl / mask.pkl / contact.pkl / intrinsics.pkl。
- 推理：`python infer_ho3d.py --ho3d-root /path/to/HO3D_v3`，默认 chunk size 128、stride ratio 0.5、2 次 refinement passes，结果在 `results/ho3d/chunk_128/re_0/`、`re_1/`。
- 评估：`python eval_ho3d.py --ho3d-root /path/to/HO3D_v3`，从预测目录加载结果，调用 `hof.evaluation.ho3d.evaluate_sequence` 计算评测指标。
- **未见完整训练入口及原始视频→初始 HOI 状态的集成流水线**；与论文所述完整方法要区别，不承诺可从原始视频直跑出全部结果。

## 实测源码调用链（静态阅读，未在 GPU 环境运行）

`infer_ho3d.py` 通过 `hof.model.builder.build_model` 构造网络并严格装载 checkpoint；`hof.inference.data.load_sequence` 加载既有初始化；`hof.inference.runner.infer_sequence` 生成重叠窗口、准备 MANO/物体几何/RGB，`infer_chunk` 调用 `model(batch)`；`WindowPredictionBlender` 合并窗口预测，多轮 refinement 输出 pickle；`eval_ho3d.py` 随后调用 `hof.evaluation.ho3d.evaluate_sequence`。

## 对 wiki 的映射

- [4D-HOF 主实体](../../wiki/entities/paper-4d-hof.md)
- [论文归档](../papers/4d_hof_arxiv_2610_08782.md)
- [官网归档](../sites/4d-hof.md)
