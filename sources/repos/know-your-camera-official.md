# Camera Conditioning 官方代码

- **Repo：** <https://github.com/ripl/CamPoseOpensource>
- **论文：** <https://arxiv.org/abs/2510.02268>
- **项目：** <https://ripl.github.io/know_your_camera/>
- **训练入口：** `policy_robosuite/train.py` 与 `policy_maniskill/train.py`；`reproduce/reproduce.py` 从 `reproduce/paper_runs.yaml` 解析实验再以 subprocess 调起训练。
- **已核对源码：** `reproduce/reproduce.py` 调用链与 `policy_robosuite/train.py` 的 Dataset/ACT/DP/SmolVLA/Evaluator 创建。
- **论文详情：** [相机几何条件策略](../../wiki/entities/paper-know-your-camera.md)
