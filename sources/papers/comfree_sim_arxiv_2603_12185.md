# ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control

> 来源归档（原始论文 · arXiv）

- **标题：** ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control
- **作者：** Chetan Borse, Zhixian Xie, Wei-Cheng Huang, Wanxin Jin（ASU IRIS Lab；UIUC）
- **arXiv：** <https://arxiv.org/abs/2603.12185>（PDF：<https://arxiv.org/pdf/2603.12185>，本次读取版本 v2，2026-03-14）
- **项目页：** <https://irislab.tech/comfree-sim/>
- **源码：** <https://github.com/asu-iris/comfree_warp>（公开；`comfree_core` 为非商业学术研究许可）
- **类型：** paper
- **入库日期：** 2026-11-06
- **一句话说明：** 用免互补（complementarity-free）的对偶锥阻抗式预测—修正，闭式求解接触冲量，接触对与摩擦锥面可解耦并映射到 GPU，运行时随接触数近线性增长；以 NVIDIA Warp 实现、兼容 MuJoCo API，可替代 MJWarp 后端。

## 对 wiki 的映射

- [paper-notebook-comfree-sim](../../wiki/entities/paper-notebook-comfree-sim.md)
- 与策展归档的关系：[humanoid_pnb_comfree-sim.md](humanoid_pnb_comfree-sim.md)（Paper Notebooks 深读笔记入口）
