# Humanoid Whole-Body Badminton via an Annealed Reinforcement Learning Curriculum（arXiv:2511.11218）

> 来源归档（ingest · arXiv 一手）

- **标题（arXiv v4，2026-09-14）：** Humanoid Whole-Body Badminton via an Annealed Reinforcement Learning Curriculum
- **别名：** 项目页与 GitHub 组织仍使用 *Multi-Stage Reinforcement Learning* 措辞
- **类型：** paper / humanoid / badminton / whole-body-control / curriculum-learning / reinforcement-learning
- **arXiv：** <https://arxiv.org/abs/2511.11218>（v4，online 2026-09-14）
- **PDF：** <https://arxiv.org/pdf/2511.11218>
- **项目页：** <https://humanoid-badminton.github.io/Humanoid-Whole-Body-Badminton-via-Multi-Stage-Reinforcement-Learning/>
- **GitHub（站点仓）：** <https://github.com/Humanoid-Badminton/Humanoid-Whole-Body-Badminton-via-Multi-Stage-Reinforcement-Learning>
- **作者（arXiv v4）：** Chenhao Liu, Leyun Jiang, Ningyuan Tian, Yibo Wang, Kairan Yao, Jinchen Fu, Xiaoyu Ren（**不含** Junzhe He）
- **机构（项目页）：** Beijing Phybot Technology Co., Ltd
- **入库日期：** 2026-09-27
- **一句话说明：** 无 MoCap/专家示范的统一全身 RL 羽毛球：**退火课程**先辅 locomotion 目标稳学习，再逐步去掉以优化击球；仿真双机 **21** 连拍；真机机喂球与人机对打，出球最高 **19.1 m/s**；EKF 与免预测变体性能相当。

## 摘要级要点

- **问题：** 羽毛球中步法与挥拍强耦合；直接优化稀疏击球奖励易梯度冲突、难收敛。
- **方法：** **Annealed curriculum** — 先用辅助 locomotion 目标稳定训练，再 **退火移除** 以聚焦最终击球目标；单策略统一 WBC，无 motion prior。
- **部署：** EKF 估计/预测羽毛球轨迹给出击球目标；另提供 **免 EKF、免显式预测** 变体（短历史球位）。
- **验证：** Isaac Gym + PPO；仿真两机器人 **21** 拍连续对打；Phybot C1 真机（1.28 m / 21 DoF）人机对打与机喂球。

## 开源核查（2026-09-27）

- **待发布：** GitHub README 仍为「All code will be released soon」；仓内仅项目站（`index.html` / `video`），**无训练/推理入口**。
- 项目页 **Code** 按钮链回项目页自身，非独立代码仓。

## 对 wiki 的映射

- [paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md](../../wiki/entities/paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md)
- [humanoid-badminton-multi-stage-rl.md](../sites/humanoid-badminton-multi-stage-rl.md)
- [humanoid_pnb_humanoid-whole-body-badminton-via-multi-stage-re.md](humanoid_pnb_humanoid-whole-body-badminton-via-multi-stage-re.md)

## 参考来源（原始）

- arXiv:2511.11218
- 项目页与 GitHub（见上）
