# ATLAS：Agentic Harness for Whole-Body Intelligence from Human Video（官方技术长文精读）

- **标题：** ATLAS: Agentic Harness for Whole-Body Intelligence from Human Video
- **类型：** official technical blog / early release（非 arXiv 正式论文）
- **作者：** Junli Ren、Kaixiang Yao、Junfeng Long、Tao Huang、Huayi Wang 等（官网列出完整 15 位作者）
- **原文：** <https://atlas-humanoid.github.io/blog/>
- **项目：** <https://atlas-humanoid.github.io/>
- **版本：** 官网标记 2026 Early release，入库核查快照 2026-11-13；未找到 PDF、arXiv ID 或 DOI
- **核对范围：** 完整阅读官方技术博客 Abstract、The ATLAS Harness、§2.1 System 2、§2.2 System 0、§2.3 fast system 的在线 model-based RL（含式 1–3）、§2.4 slow system 的 agentic update、§2.5 retry in place、参考文献、BibTeX；对照官方主页 Overview、Method、Learning、Tasks、Evaluation、Setup & Scope。当前公开页面没有完整独立论文 PDF，后续发表应另行复核。
- **通读：** 已完成
- **代码核查：** 2026-11-13 项目官网/技术博客未列 ATLAS 官方可运行源码链接；不以 ScaleBFM 上游仓库充当 ATLAS 源码。

## 主要机制（按原文）

1. **人视频到参考动作：** 从一段人-物互动视频恢复人的动作并 retarget 到人形机器人；VLM 根据提示词及动作分段，生成含 step、末端、接触 affordance、分割提示和完成条件的任务规格。有限枚举字段映射到固定行为。
2. **几何接地：** SAM 3 对机器人头部相机图像按 find phrase 分割物体，深度恢复目标 3D 位置；将参考动作中的 reach 终点对齐物体。导航由规划器给出轨迹；开始 pick 后换为重定向参考，但**冻结 tracker 本身不切换策略**。
3. **System 0：** 冻结的 ScaleBFM 全身跟踪策略以 50 Hz 读取运动窗口，输出关节动作；接触过程为 crouch、reach、hold（0.5 s）、close、lift。
4. **fast system：** 在线 actor 根据末端实际接触点 `c_t` 和目标 `n_t` 产生残差 `u_t = πφ(c_t,n_t)`，tracker 接收 `n_t + u_t`。随机先验 ensemble 动力学模型 `ĉ_(t+1)=fψ(c_t,n_t+u_t)` 用真实尝试重拟合；actor 对模型 rollout 的 hold 误差 `-||n_t-ĉ_t||²` 及残差变化惩罚反传。无需 critic 或成功标签；每会话从零开始，前五次探索热身。
5. **slow system：** VLM 不处于 50 Hz 闭环里；发现误分割/错误的成功判断后修改 find phrases 或 done checks，必须在保存的 held-out 帧/attempt 上回放比较，只有召回改善且误匹配不增等条件满足才采用。
6. **retry in place：** 闭爪未抓到、未确认抬起或掉落即判失败；机器人原地起身、重新观察并重试，错误样本持续进入在线训练。人员不遥控，但在成功后/物体出视野时可能重置物品。

## 官方结果与统计口径

- **真机在线适应：** 官方首页展示一个拾取垃圾 session 内 38 次尝试，最近 8 次窗口成功率从 0% 到 88%；单任务案例不等于所有任务的平均提升。
- **对任务专用 GR00T checkpoint：** 技术博客作者报告真实场景中 loco-manipulation 和静态 pick-and-place 成功率分别高 36.2% / 12.6%（原文用 higher success rates，未给出此处完整样本数与绝对/相对口径）；不可直接改写成百分点。
- **Isaac Lab-Arena：** 技术博客报告 ATLAS loco-manipulation 94% vs GR00T 69%、静态 pick-and-place 98% vs 87%，是其指定任务/benchmark 下的对照。
- **动态误差模型消融：** 官方首页给出 apple、变化偏差设置，模型式 RL 59/71=83%，RLS 偏差估计 66/114=58%，model-free 9/71=13%，无学习 16/378=4%；每个算法实际分到的回合数量不同，不能用作独立复现证据。

## 局限、开放问题

- 目标是短时接触与末端对准；官网将持续接触、跨任务保留真实世界经验列为未来工作，说明这些尚未证明。
- 人视频恢复与重定向依赖上游算法和冻结的 ScaleBFM；“不依赖任务专用训练”不等于没有任何训练过的基础模型。
- slow VLM、SAM 3、学习器更新在外部 PC，经通信链接接入机载 actor/tracker；部分平地任务仍有外部定位。
- 项目主页未给源码/权重/完整独立 PDF；实验为作者报告，无法复现代码级时序；未知真机实验分母、故障率与统计显著性不能补猜。

## 对 wiki 的映射

- [ATLAS：代理规格与在线残差连接的全身操作框架](../../wiki/entities/atlas-agentic-harness.md)
- [项目主页归档](../sites/atlas-humanoid.md)
