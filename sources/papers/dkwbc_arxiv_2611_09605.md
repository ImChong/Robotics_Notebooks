# DKWBC：解耦动作跟踪与平衡的人形全身控制（arXiv:2611.09605）

> 来源归档（paper）

- **标题：** DKWBC: Decoupling Motion Tracking and Balancing for Humanoid Whole-Body Control
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.09605>（v1，2026-11-06，cs.RO）
- **HTML 全文：** <https://arxiv.org/html/2611.09605v1>
- **PDF：** <https://arxiv.org/pdf/2611.09605v1>
- **项目页：** <https://robbyant-research.github.io/DKWBC/>
- **官方仓库：** <https://github.com/robbyant-research/DKWBC>（当前为 gh-pages 项目展示仓库）
- **作者：** Dingkun Wei, Zehong Shen, Huaijin Pi, Xueyang Wang, Xing Zhu, Hujun Bao, Yujun Shen, Xiaowei Zhou
- **机构：** 浙江大学（Zhejiang University）、蚂蚁灵波（Robbyant）、香港大学（The University of Hong Kong）
- **入库日期：** 2026-11-10
- **一句话说明：** 用浮基逆动力学前馈与关节 PD 负责参考运动跟踪，让 RL 策略只输出补充的平衡力矩，以降低动作跟踪与平衡在同一策略中耦合的难度。

## 核对范围与通读状态

- **核对范围：** 通读 arXiv v1 HTML 完整论文的摘要、引言、相关工作、III 方法（式 1–7、图 2）、IV 实验（数据规模、外力跟踪、解耦及组件消融，表 1–4、图 3–6）、结论与参考文献；结合官方项目页核对 G1 演示、SONIC 数据缩放实验、项目 Code 链接，并检查官方 GitHub 仓库 README 与默认分支。HTML 版本与论文 v1 对应；未读取未公开训练代码。
- **通读：** 已完成

## 原文核心摘录（按论文结构）

1. **动力学分解（III-A，式 1–2）：** 浮基动力学为 M(q)·ν̇ + h(q,ν) = Sᵀτ + Σ Jₖᵀλₖ。将致动关节力矩分为可用模型预测的运动动力学项，以及难以直接测量的环境接触/外力项；此分解是控制器设计动机，**不表示残差策略直接识别了真实接触力**。
2. **显式跟踪（III-B，式 3–6）：** 参考速度先差分，再经三帧高斯平滑（σ=0.75 帧）求参考加速度；浮基逆动力学使用 recursive Newton–Euler 算法，得到 τ_id = τ_acc + τ_cor + τ_grav；加关节位置和速度 PD 反馈 τ_pd，合成 τ_trk。
3. **学习式平衡（III-C，式 7）：** actor 以参考动作、可部署本体信息、τ_acc/τ_cor/τ_grav 作为观测，直接输出 50 Hz 关节残差力矩，按每关节力矩上限的 0.25 缩放并在两个策略更新之间保持；τ_cmd = τ_id + τ_pd + τ_bal。训练采用模仿跟踪的 RL 框架，actor 观测借鉴 BeyondMimic；与传统位置目标策略相比，不把残差映射为快速 PD 位置伺服。
4. **数据缩放（IV-A，图 3–4）：** BONES-SEED 清洗后 128.53 小时训练动作，采样 2/4/8/16/32/64/128 h；test-content 为 3,145 条 / 6.031 h，test-repetition 为 3,019 条 / 6.002 h。对比重训 SONIC（去掉 VR 与 SMPL 输入），各训练规模 DKWBC 成功率更高、MPJPE 更低；论文图 3 为多检查点结果，不应捏造一个全规模准确百分比。
5. **外力实验（IV-B，表 1）：** 从 G1 重定向 LAFAN 与 AMASS Test 选 217 条（209 移动 + 8 站立）。三档手部外力 α_g=0/0.5/1；最大外力下 DKWBC 上身 MPJPE 24.68 mm、MPJVE 0.12 m/s、root 速度误差 0.16 m/s、成功率 99.54%；FALCON 同档 83.51 mm / 99.08% 成功率，ResMimic 54.96 mm / 100% 成功率。精度优于所有所测基线，但成功率并非最高。
6. **显式解耦（IV-C1，表 2）：** 与匹配网络、奖励、训练预算的 BeyondMimic coupled 基线比较，LAFAN 无额外载荷下 DKWBC 成功率 98.95% 对 97.67%，MPJPE 26.59 对 34.06 mm；BONES-SEED test-content 扰动下 92.18% 对 90.30%，MPJPE 33.65 对 42.07 mm。载荷设置为 5 kg 背包以及各手 2 kg，扰动通过 root 速度施加。
7. **跟踪模块与消融（IV-C2/C3，表 3–4）：** 无外载荷的纯跟踪比较中 ID+PD 4.45 mm MPJPE，对单纯 PD 原增益 63.19 mm、单独 FF 为 171.73 mm；**此指标属于单独跟踪模块实验，不等同完整全身实机误差**。表 3 完整 DKWBC 在 BONES-SEED test-repetition 90.86% 成功率、43.16 mm MPJPE；删去 PD 后为 88.64% / 50.82 mm。去掉动力学观测某些配置在训练集上变好、在未见测试集上反而变差。
8. **真实机器人（IV-B 图 5、项目页）：** Unitree G1 同一策略 checkpoint 在真机演示深蹲、载荷下行走和复杂动作；论文未公布与 SONIC 的真机定量对照，也没有真机大规模统计置信区间。
9. **局限与未来（V）：** 动作运动学参考若动力学不可行，纯前馈效果差，必须依赖反馈和残差；训练动作曾人工排除不符合实验设置的动作，跨机器人型号、任意高接触任务与长期负载并无系统验证；作者拟扩展全身物理交互任务。

## 开源状态（2026-11-10 核查）

- 官方项目页有 Code 链接，指向 <https://github.com/robbyant-research/DKWBC>；仓库默认分支为 `gh-pages`，README 仅说明“Official project page for Decoupling Motion Tracking and Balancing for Humanoid Whole-Body Control”。
- **已公开项目展示源码，但未见可运行的 DKWBC 训练 / 推理 / 部署实现、权重或明确运行入口。** 不应把 Code 按钮等同可复现算法。
- 对 wiki 的源码时序图：**不适用**，无可核实运行时入口；若后续发布实现须再读代码并补图。

## 对 wiki 的映射

- [DKWBC：动力学跟踪与学习式平衡解耦](../../wiki/entities/paper-dkwbc.md)
