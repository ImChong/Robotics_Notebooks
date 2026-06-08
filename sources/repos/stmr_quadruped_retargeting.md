# STMR 四足时空重定向生态

> 来源归档（三仓库一组）

- **论文：** Spatio-Temporal Motion Retargeting for Quadruped Robots（arXiv:2404.11557）
- **入库日期：** 2026-06-08
- **一句话说明：** 将动物/噪声关键点轨迹经 **空间重定向（SMR）+ 时间重定向（TMR）** 转为四足全身可跟踪参考，再接 legged_gym RL 训练。
- **沉淀到 wiki：** 是 → [`wiki/entities/stmr-quadruped-retargeting.md`](../../wiki/entities/stmr-quadruped-retargeting.md)

## 仓库矩阵

| 仓库 | 链接 | 职责 |
|------|------|------|
| Quadruped_Retargeting | https://github.com/terry97-guel/Quadruped_Retargeting | 空间运动重定向 SMR |
| Quadruped-Motion-Timing | https://github.com/terry97-guel/Quadruped-Motion-Timing | 时间运动重定向 TMR |
| STMR_RL | https://github.com/terry97-guel/STMR_RL | 完整 RL 训练管线（Go1/A1/Aliengo 等） |
