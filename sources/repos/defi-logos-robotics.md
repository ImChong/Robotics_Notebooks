# defi-logos-robotics

> 来源归档（ingest）

- **标题：** LogosRoboticsGroup/DeFi — Disentangled Robot Learning 官方实现
- **类型：** repo
- **链接：** <https://github.com/LogosRoboticsGroup/DeFi>
- **权重：** <https://huggingface.co/zbzzbz/DeFI>
- **论文：** <https://arxiv.org/abs/2604.16391>
- **入库日期：** 2026-05-20
- **一句话说明：** DeFI（GFDM + GIDM + 扩散动作适配器）的训练/评测代码与 checkpoint 发布入口，对应 arXiv:2604.16391。

## 仓库要点（README 级）

- **模块边界：** 前向动力学（视频生成 / SVD 系）、逆动力学（自监督潜动作 + VQ）、下游耦合微调与 CALVIN / SimplerEnv / 真机评测脚本（以仓库当前目录为准）。
- **依赖生态：** 与论文一致的扩散视频骨干、DINOv2、T5 指令编码、DiT 动作适配器等；具体环境与数据准备见官方 README。
- **复现注意：** 多视角（第三人称 + 腕部）在 CALVIN 设定下与论文 Table 1 两行对照相关；真机 Franka 八任务协议见论文 §4.4。

## 关联 Wiki 页面

- [DeFI（解耦前向/逆动力学 VLA）](../../wiki/methods/defi-decoupled-dynamics-vla.md)
- [VLA](../../wiki/methods/vla.md)

## 当前提炼状态

- [x] 官方链接与论文映射
- [ ] 后续：随仓库 README 更新补充训练命令与 checkpoint 命名

## 固定版本源码复核（2026-11-13）

- 原LogosRoboticsGroup/DeFi已重定向[LogosIntelligenceLab/DeFi](https://github.com/LogosIntelligenceLab/DeFi)，固定版本[3f8666c43a19edb1a25b7b50210af95699803708](https://github.com/LogosIntelligenceLab/DeFi/tree/3f8666c43a19edb1a25b7b50210af95699803708)。已读README、Step3_DeFI/scripts/rollout_calvin.sh、wrappers/hulc_wrapper.py全文，calvin_evaluate.py的main/evaluate_policy/evaluate_sequence/rollout，VPP_policy.py的eval_forward/step/denoise_actions完整函数。
- rollout_calvin.sh用Accelerate启动calvin_evaluate，Hydra实例化策略、数据和HulcWrapper；rollout调用VPP_Policy.step，按replan间隔eval_forward：双视角SVD特征→对齐与GIDM→denoise_actions；缓存chunk后env.step、task_oracle判成功。
- README区分Step1 GFDM、Step2 GIDM、Step3 DeFI，CALVIN公开实现可读；未证明Simpler/真实机器人入口全覆盖，未下载权重或跑GPU复现。
