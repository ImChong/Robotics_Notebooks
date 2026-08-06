# X-Morph（arXiv:2606.30290）

> 来源归档（ingest）

- **标题：** X-Morph: Human Motion Priors for Scalable Robot Learning Across Morphologies
- **类型：** paper / cross-morphology / motion-retargeting / motion-prior / legged / loco-manipulation / RL
- **arXiv abs：** <https://arxiv.org/abs/2606.30290>
- **PDF：** <https://arxiv.org/pdf/2606.30290>
- **HTML：** <https://arxiv.org/html/2606.30290>
- **项目页：** <https://maker-rat.github.io/morph/> — 归档见 [`sources/sites/maker-rat-morph-github-io.md`](../sites/maker-rat-morph-github-io.md)
- **机构：** 新加坡国立大学（NUS / National University of Singapore）；资助声明含 Singapore MOE 与 NUS Robotics Grand Challenge
- **作者：** Ritwik Sharma*†、Shivam Sood*、Arhaan Jain、Shyam Charan Kesavamoorthi、Chengyang He、Guillaume Sartoretti（* equal contrib.；† corresponding）
- **发表 / 上传：** 2026-06-29（arXiv v1）
- **平台：** Unitree Go2（四足）、Yuna（六足）、B2-Z1（四足 + Z1 机械臂）；源运动先落到 Unitree G1 表示
- **入库日期：** 2026-08-06
- **一句话说明：** 把大规模人体运动经「跨形态重定向 → 物理感知校正 → 特权 RL 跟踪 → 因果学生蒸馏」转成非人形腿式机器人可部署 locomotion / loco-manipulation 行为先验，并接视频遥操作与文本条件技能执行。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv | [2606.30290](https://arxiv.org/abs/2606.30290) | 论文与附录 |
| 项目页 | [maker-rat.github.io/morph](https://maker-rat.github.io/morph/) | 演示与入口；Code/Video 按钮仍 disabled |
| 重定向底座 | PAN (Hu et al. 2024) | body-part 编解码 + pose-aware attention |
| 部署前端 | [GMR](https://arxiv.org/abs/2510.02252) / FastSAM3D Body / Kimodo | 视频→SMPL→G1；文本→G1 运动 |

## 开源状态（步骤 2.5，截至 2026-08-06）

- **宣称将开源 / 未列链接：** 项目页有 **Code** 与 **Video** 按钮，但均为 `disabled`（`href="#"`）；公开材料**无** GitHub / Hugging Face URL。
- **处理：** wiki 按「截至入库日无可运行官方代码」写；`## 源码运行时序图` 标不适用。后续若放出 URL，补 `sources/repos/` 并互链本页。

## 摘要级要点

- **问题：** 人形行为模型吃到大量人体运动数据；四足 / 六足 / 带臂四足等非人形腿式机器人缺少同量级行为库。直接跨拓扑重定向常「看着像」但脚滑、穿地、漂浮接触、难跟踪。
- **主张：** 把人体运动当成**可复用行为底物**，而不是最终控制器；重定向只产出候选参考，必须经物理校正 + 闭环跟踪才可部署。
- **管线：** 人体/G1 源运动 → 形态感知重定向 \(f_\theta\) → 物理感知校正 \(c_\psi\) → 特权 teacher 跟踪 + 因果 student 蒸馏；交互部署另训因果重定向学生 \(g_\eta\)。
- **平台：** Go2、Yuna hexapod、B2-Z1；任务族含 locomotion 与 loco-manipulation（手臂↔前肢/Z1 臂语义对应表见 Appendix A）。
- **下游接口：** 视频遥操作（≤28.9 Hz 参考流）、文本→Kimodo→G1→同一跟踪栈、下游开门等任务初始化先验（定性，非 vs-scratch 对照）。
- **关键读数：** Go2 参考消融（Table 1）corrector 使 foot slip −27.2%、penetration −46.9%；Yuna 跟踪（Table 2）corrected refs 使 Joint MAE −17.4%、yaw-rate RMSE −27.5%。

## 核心摘录（面向 wiki 编译）

### 1) 问题分解：参考生成 + 参考跟踪

\[
\hat{\mathbf{s}}^{r}=f_{\theta}(\mathbf{x}^{h},\mathcal{M}^{r},\mathcal{C}^{h\rightarrow r}),\quad
\tilde{\mathbf{s}}^{r}=c_{\psi}(\hat{\mathbf{s}}^{r},\mathbf{x}^{h},\mathcal{M}^{r},\mathcal{C}^{h\rightarrow r}),\quad
a_{t}\sim\pi_{\phi}(a_{t}\mid o_{t},\tilde{s}^{r}_{t:t+H}).
\]

目标不是字面模仿人体，而是保留意图 / 时序 / 肢体协调 / 接触结构，并允许形态特有偏差。

### 2) 形态感知重定向（PAN + 机器人损失）

- 人体 AMASS / LAFAN1 等先表示到 **Unitree G1**，再跨形态学映射。
- 目标帧：\(m_t^r=[q_t^r,\,v_t^r,\,\omega^r_{z,t}]\)（MuJoCo 关节序 + 局部根速度 + 偏航率）。
- 损失：PAN 式重建 / cycle / adversarial / FK + **脚滑、接地、穿地、关节限位、操作末端对齐**。

### 3) 物理感知校正与跟踪蒸馏

- Corrector：离线时间残差 CNN，编辑关节与根轨迹，压接触伪影。
- Teacher：特权全状态 + APEX-style action prior + DeepMimic 式跟踪奖励。
- Student：仅可部署本体感觉 + 短参考预览；\(\mathcal{L}_{\mathrm{distill}}=\|\pi_\phi-\pi_{\theta^\star}\|_2^2\)。
- 因果重定向学生：用源 G1 短历史 + 自回归目标上下文模仿「校正后参考」，供在线视频/交互。

### 4) 主结果（Table 1 / 2）

| 场景 | 关键相对变化（corrector vs raw） |
|------|--------------------------------|
| Go2 参考质量（33 locomotion clips） | slip −27.2%；penetration −46.9%；contact height err. −44.2%；floating −39.3% |
| Yuna 直播视频参考跟踪 | Joint MAE −17.4%；root vel RMSE −13.9%；yaw-rate RMSE −27.5%；foot slip −17.5% |

### 5) 局限（论文 §6）

- 依赖**手工语义对应** \(\mathcal{C}^{h\rightarrow r}\)；对应差则难校正/跟踪。
- Corrector **不是**完整轨迹优化，不保证动力学可行。
- 评测以平坦/中等结构地形为主；视频链路受单目姿态估计质量与延迟约束。

## 对 wiki 的映射

- 新建实体页：[wiki/entities/paper-xmorph.md](../../wiki/entities/paper-xmorph.md)
- 交叉：[hub-motion-retargeting](../../wiki/overview/hub-motion-retargeting.md)、[hub-cross-embodiment](../../wiki/overview/hub-cross-embodiment.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md)、[ReActor](../../wiki/methods/reactor-physics-aware-motion-retargeting.md)、[ZEST](../../wiki/methods/zest.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[Unitree](../../wiki/entities/unitree.md)
