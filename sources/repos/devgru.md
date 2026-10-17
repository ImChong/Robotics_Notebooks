# EwhaGlab/devgru

> 来源归档（ingest · 代码仓库）

- **标题：** DevGRU
- **类型：** repo / visual navigation / point-goal navigation
- **链接：** <https://github.com/EwhaGlab/devgru>
- **论文：** [arXiv:2608.18470](https://arxiv.org/abs/2608.18470)；IEEE RA-L 2026
- **项目页：** [graphics.ewha.ac.kr/projects/details/devgru](https://graphics.ewha.ac.kr/projects/details/devgru/)
- **代码许可证：** 自定义限制许可：仅教育、研究及非营利用途可免费使用、复制、修改或分发；商业用途须另行取得许可。
- **权重 / 数据：** README 链接预训练权重与 mixed_20K 数据集下载。
- **入库日期：** 2026-10-17
- **一句话说明：** 官方实现包含 DevGRU 的 AP / CP 训练代码与 ROS 部署脚本，以深度和拓扑点目标完成室内长程导航。
- **沉淀到 wiki：** [paper-devgru](../../wiki/entities/paper-devgru.md)、[NoMaD 对照](../../wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md)、[NavDP 对照](../../wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md)

---

## 仓库结构与复现路径

| 路径 / 入口 | 作用 |
|---|---|
| `train/` | AP、CP 模型训练；通过 `config/devgru.yaml` 配置数据路径 |
| `train/train_action_predictor.py` | 训练 Action Predictor |
| `train/train_collision_predictor.py` | 训练 Collision Predictor |
| `deployment/src/` | ROS 真机部署脚本；README 用 `run_navigate_viz.py` 启动导航 |
| NavDataCollector | 采集 ROS bag、从 bag 构建拓扑图，提供部署数据支持 |
| `train/logs`、`deployment/model_weights` | 训练日志与部署模型权重目录 |

## 工程注意

- README 环境为 Ubuntu 20.04、ROS Noetic、Conda；训练和部署分别使用独立环境。
- 复现需录制 ROS bag，再用 NavDataCollector 提取同步数据并生成拓扑地图；导航时连接 ROS master 后运行部署脚本。
- 仓库继承 ViNT 的部分实现。代码公开不等于可商用：LICENSE 限教育、研究及非营利用途。
- 部署说明以 FORMER 轮式移动机器人为主；不要将校园场景结果直接外推至腿式或人形平台。
