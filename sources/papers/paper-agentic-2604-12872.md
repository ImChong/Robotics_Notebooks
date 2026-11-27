# OVAL: Open-Vocabulary Augmented Memory Model for Lifelong Object Goal Navigation

- **作者：** Jiahua Pei*, Yi Liu*, Guoping Pan, Yuanhao Jiang, Houde Liu†, Xueqian Wang†（* 同等贡献；† 通信作者）
- **机构：** 清华大学深圳国际研究生院（Shenzhen International Graduate School, Tsinghua University）
- **版本 / 载体：** arXiv:2604.12872v1 [cs.RO]（2026-04-14 提交；8 页，5 图，4 表）
- **原文：** <https://arxiv.org/html/2604.12872>（摘要页：<https://arxiv.org/abs/2604.12872>）
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，论文原文及公开索引均未提供官方项目页或 GitHub 代码仓库链接）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§I Introduction & Related Work、§II Overview & Algorithm 1、§III Implementation、§IV Experiments & Table I–IV、§V Conclusion and Limitations、References）
- **通读：** 已完成

## 原文核对笔记

1. **问题定义（§II-A）：**
   - 将单目标 ObjectNav 扩展为 **Lifelong ObjectNav**：智能体在未知环境中连续接收开放词汇物体目标序列 $\hat{G}_t$（如 `cup -> remote control -> ...`），下一目标在上一目标完成后动态下发（区别于预定义列表的 MultiON）。
   - 观测 $O_t=(V_t,D_t,P_t)$ 包含 RGB、深度和位姿；动作空间为 `{Stop, Move Forward, Turn Left, Turn Right, Look Up, Look Down}`；在 500 步内距目标 1 m 内触发 `STOP` 判定为成功。
2. **三大核心模块（§II-B、§III）：**
   - **Frontier Exploration（§III-A）：** 由深度和位姿投影构建 2D 俯视障碍栅格图并聚类 frontier；记录已尝试 frontier 避免来回振荡；以多值概率评分 $P(F)\propto o_d(F)+o_s(F)+o_f(F)$ 综合距离、共现语义先验与历史轨迹足迹惩罚选择探索点。
   - **Open-Semantic Memory Model（§III-B）：** 使用 Grounded-SAM2（`Autolabel` 模式，搭配 Florence-2-base 的 `<DETAILED_CAPTION>`）生成开放词汇检测，并用关键词过滤 `"wall"`、`"various objects"` 等噪声标签；初始化阶段用 GPT-4o-mini 生成目标同义词表 $\overline{G}_t$。记忆条目表示为 $\mathcal{C}=\{S_i,\mathcal{I}_i,X_i,H_i,C_i\}$（标签、图像缓存、3D 平面位置、HSV 场景直方图、置信概率 $C_i$）。新观测与已有同名记忆通过位置+HSV 相似度 $S_m$ 粗筛，模糊区间 $(\tau_l,\tau_u)$ 调用 SuperGlue 局部特征匹配；确认同一实例且 $C_i \le C_j$ 时用高置信观测覆盖更新，避免冗余存储与记忆漂移。
   - **Navigation & Verification（§III-C）：** 用 KMP 子串匹配实现跨粒度标签检索（如用 `"desk"` 命中 `"green desk"`），调用贪心 PointNav 路径规划器抵达 $X_i$ 后，采集 360° 全景图送入 GPT-4o 计算到点验证概率 $\phi$，按 $S=\omega\phi+(1-\omega)C_i$ 决定触发 `STOP` 或剔除错误记忆继续探索。
3. **数据集构建与实验结果（§IV）：**
   - **Lifelong 数据集生成（§IV-B）：** 将 HM3D / MP3D 验证集按 Scene ID 与楼层高度（差值 $<0.5\text{ m}$ 视为同层）重排分组；同组内保留地图与记忆并瞬移至上一集终点，跨组重置。
   - **Lifelong ObjectNav（Table I，1000 episodes）：** HM3D 上 OVAL 达 SR 68.1% / SPL 33.8%，高于 GOAT（59.2% / 31.2%）与 VLFM（53.7% / 30.6%）；MP3D 上 OVAL 达 SR 44.1% / SPL 18.6%，高于 VLFM（35.1% / 16.7%），而 GOAT 因闭集检测器限制运行失败（N/A）。运行帧率约 1.14 FPS（对比 VLFM 1.67 FPS、GOAT 1.85 FPS）。
   - **消融实验（Table II、Table IV）：** 去掉 Memory Model 后 HM3D Lifelong SR 从 68.1% 降至 56.1%（SPL 33.8%→26.0%）；去掉 Verify `"STOP"` 降至 61.3% / 31.2%；去掉 Probability Map 降至 66.8% / 32.3%。在单次 ObjectNav 的概率图消融中，去掉足迹惩罚 $o_f(F)$ 对 SR/SPL 影响最大（58.2%/24.5% → 54.3%/21.4%）。
   - **标准 ObjectNav（Table III，1000 episodes）：** HM3D 上 OVAL 达 SR 58.2% / SPL 24.5%（无训练方法对比中 SR 最高）；MP3D 上达 SR 41.1% / SPL 15.3%。
4. **原文局限与核对结论（§V）：**
   - 仅支持 ObjectNav 物体目标输入，尚不支持多模态复杂指令；在动态、复杂、快速移动场景中识别准确率会下降；语义共现表需预定义，且同义词扩展与终局核验依赖 GPT-4o / GPT-4o-mini。
   - 对 wiki 的映射：[OVAL 实体页](../../wiki/entities/paper-agentic-2604-12872.md)
