# GenMimic: From Generated Human Videos to Physically Plausible Robot Trajectories（arXiv:2512.05094）

> 来源归档（2026-11-20 补充 arXiv 原文全读）

- **标题：** GenMimic: From Generated Human Videos to Physically Plausible Robot Trajectories
- **作者：** James Ni 等（UC Berkeley、NYU、JKU）
- **arXiv：** <https://arxiv.org/abs/2512.05094>（`2512.05094v2`）
- **项目页：** <https://genmimic.github.io/>
- **评测基准：** <https://huggingface.co/datasets/wlin21at/GenMimicBench>
- **代码：** 截至 2026-11-20 仍未开源（官网标注 coming soon）
- **类型：** paper
- **入库日期：** 2026-11-20
- **通读：** 已完成
- **核对范围：** 2026-11-20 通读 `arXiv:2512.05094v2` 论文全文（`1–1932` 行），核对 Stage 1（TRAM 4D 人体重建 + PHC 重定向为三维关键点轨迹 `3DP`）与 Stage 2（Isaac Gym 中基于 `8,123` 条 AMASS 过滤动作的 PPO 教师 → DAgger `MLP [512, 256, 128]` 学生蒸馏，50 Hz 控制 / 200 Hz PD）、三维关键点条件（`3DP`）、部位分层加权奖励（`w_end-eff=4, w_upper=2, w_lower=1`）与双边对称损失（`L_SYM`）、`GenMimicBench` 428 条生成视频基准（217 Wan2.1 NTU + 211 Cosmos-Predict2 PennAction）的教师 `86.77%` / 学生 `29.78%`（vs GMT `4.29%` / TWIST `7.52%`）对比及 23-DoF Unitree G1 吊架真机部署（43 条动作：原地 VSR `1.0`，移步/转身 `~0.40`）。

## 核心机制与定量结果

1. **为什么不用全关节点强跟踪？** 视频生成模型（Wan2.1 / Cosmos-Predict2）产出视频经 TRAM 重建后常出现几何缩放漂移、脚滑与物理失真；GenMimic 改为仅传全局三维关键点位置特征（`3DP`）而非脆弱的多模态完整参考，把下肢与平衡细节交由仿真 RL 补全。
2. **三项训练支柱：** ① 全局 **3D Point Conditioning (`3DP`)**；② **部位分层加权奖励**（`w_end-eff=4` 优先手腕/头/关键末端，`w_upper=2`，`w_lower=1` 允许下肢灵活调整重心与落脚）；③ **左右双边对称正则** `L_SYM` 抑制策略偏斜。
3. **跨视频模型迁移与真机边界：** 在 `GenMimicBench`（428 条合成视频）上教师策略成功率 `86.77%`、真机可部署学生策略 `29.78%`（显著高于 GMT `4.29%` 与 TWIST `7.52%`）；但 23-DoF G1 吊架真机测试显示：原地任务 VSR 达 `1.0`，而大步幅位移/转身跌至 `~0.40`，暴露学生仅靠关键点位置推理动态步态与接触稳定性仍有差距。

## 对 wiki 的映射

- 论文实体页：[paper-hrl-stack-04-from-generated-human-videos-to-physi](../../wiki/entities/paper-hrl-stack-04-from-generated-human-videos-to-physi.md)
- 项目页归档：[genmimic](../sites/genmimic.md)
- 42 篇栈导读归档：[humanoid_rl_stack_04_from_generated_human_videos_to_physically_plausi](./humanoid_rl_stack_04_from_generated_human_videos_to_physically_plausi.md)
