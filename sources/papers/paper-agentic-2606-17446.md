# AnnotateAnything: Automatic Annotation of 3D Assets for Robot Manipulation

- **作者：** Haoran Lu, Mutian Shen, Shuyang Yu, Yu Xiao, Songling Liu, Jianshu Zhang, Shang Wu, Yue Chen, Guo Ye, Jiayi Wang, Zhaoran Wang, Han Liu
- **机构：** Northwestern University、中国科学院自动化研究所（CASIA）、中国科学院大学（UCAS）、北京大学（PKU）
- **版本 / 载体：** arXiv:2606.17446v2 [cs.RO]（2026-06-16 提交 v1，2026-08-30 修订 v2，arXiv HTML 全文）；项目页（NeurIPS 2026 匿名投稿展示页）
- **原文：** https://arxiv.org/abs/2606.17446 / https://arxiv.org/html/2606.17446 / https://tourmaline-caramel-169490.netlify.app
- **项目页 / 代码：** 宣称将开源或未列链接（项目页标注 "Code Coming Soon"，论文摘要及 Appendix J.5 说明代码、演示资产与标注已放入匿名投稿补充材料并计划后续公开发布，截至 2026-11-27 未公开独立可克隆的官方代码仓库；论文参考文献 §I.4 引用的 `https://github.com/benelot/pybullet-gym` 为第三方仿真环境，非本论文官方仓库）
- **核对范围：** 通读 arXiv HTML v2 全文（Abstract、§1–§6、References 及 Appendix A–J 全部 3,419 行，含 Tables 1–43）及项目页正文（50 行）；核查 candidate_repos。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[AnnotateAnything 实体页](../../wiki/entities/paper-agentic-2606-17446.md)

1. **核心定位与解决的四类挑战（§1）：**
   - 现有 raw 3D 资产仅含几何与外观，缺乏语义、交互与物理约束信息。AnnotateAnything 旨在解决：C1（融入人类交互先验）、C2（将抽象意图落地为资产特定几何与运动学可执行标签）、C3（保留多解多样性而非坍缩到单一规范解）、C4（大规模自动化，无需逐资产手工设计）。
   - 覆盖四类资产（刚体、铰接体、柔性/衣物、房间级导航场景）及 18 个原子技能（归纳为 Grasp、DexGrasp、BiGrasp、BiDexGrasp、Articulation、Insertion、Hanging、Deformable、Navigation / Approach 九大技能族）。
2. **层级视觉-语言标注管线（§3.1, Appendix C–D）：**
   - **物体级语言（§3.1.1, C.1）：** 围绕包围盒归一化渲染 8 视角 RGB，输入 `Qwen3.5-VL` 生成三层描述（phrase / sentence / paragraph），并对高亮分割部件逐一查询生成结构化 JSON。
   - **房间级语言（§3.1.1, C.2）：** 采样覆盖多实例的 8 个房间视角 + 带标签俯视占据图输入 `Qwen3.5-VL`，生成短句、长句与段落级布局/交互描述。
   - **物体级视觉（§3.1.2, D.1）：** 8 视角 RGB-D 反投影融合点云 $\mathcal{P}_{cam}$，用网格参考点检查覆盖度并补充视角，FPS 下采样至 $\mathcal{P}_{80k}$，经 182M 参数 Hunyuan3D 部件分割模型（基于 P3-SAM / X-Part）得到部件掩码，并由 VLM 锚定关键点与可供性区域。
   - **房间级视觉（§3.1.2, D.2）：** 按墙体分解房间，贪心选视角最大化可见实例覆盖，用多高度 LiDAR 射线构建占据栅格 $\mathcal{G}_j$，拼接全局图并用 VLM 过滤可移动家具以保留永久墙体结构。
   - **一致性校验：** 解析失败或无法关联有效分割 ID 的输出被拒绝（3,240 次查询中拒绝 38 + 56 = 94 个可检测失败项，占比 2.9%）。
3. **物理动作标注管线（§3.2, Appendix E–F）：**
   - **统一候选库 Schema（§3.2.1, E.1）：** 定义一到多映射 $\mathcal{B}_{h,s} = \{a_{h,s}^{(i)}\}$，每条候选包含技能 $s$、物体 $o(h)$、锚点 $h$、功能可供性 $\phi(h)$、目标 $x^{(i)}$、参数 $\theta^{(i)}$、轨迹 $\tau^{(i)}$、验证元数据 $v^{(i)}$ 与多样性描述 $d^{(i)}$。
   - **五阶段流水线（§3.2.2, E.2–E.6）：** 候选目标生成（FPS/关键点/占据图采样 + 几何/可达性过滤）→ 四类约束模板生成初始轨迹（物体属性、物体轨迹、物体向量、场景轨迹）→ 轨迹/位姿优化（L-BFGS-B 解 5mm SDF 灵巧手多阶段目标、几何自适应关键点缩放、铰接体追踪、A*+DWB 导航优化）→ 并行物理验证（Headless Isaac Sim，物体中心技能用浮动夹爪/灵巧手加 $K=8$ 组 $1.5g_0$ 随机重力扰动，衣物/强本体依赖技能用完整机械臂，移动导航用基底感知验证）→ 物理感知增强（局部扰动与对称变换后再验证）。
4. **下游数据采集与应用（§4, Appendix G）：**
   - 构建基于 `cuRobo-v2` 异步规划与 goal-set IK 的大规模并行仿真采集系统（域随机化后按排序从候选库回退检索可执行候选）。
   - 支持感知监督替换（PartNeXt/SAMPart3D: 43.9 vs 46.70 mIoU；LASO: 40.7/54.1 vs 43.6/57.2；KeypointNet/Skeleton Merger: 74.5/73.6 vs 77.7/76.8 DAS），以及作为示例展示的 Robot Reasoning VQA 与 3D VLM 指令微调数据构建。
5. **核心实验结论与局限（§5, Appendix H, Appendix J）：**
   - **规模与转换（§5.2, H.2）：** 处理 9 大来源族共 17,005 个资产，产出 100M 物理验证动作实例（18 个原子技能）。在 1,160 资产 / 3,698 资产-技能对的审计集上，平均每对生成 2,315 候选 → 1,384 几何可行 → 842 轨迹可行 → 615 物理验证通过（26.6%）→ 538 保留（23.2%）。
   - **视觉-语言与动作评测（§5.3–5.4, H.3–H.4）：** 360 人工参考集上完整 VL 包人类评分 90.9±0.6（显著高于最强消融 w/o 3D refine 的 78.5±1.2）。审计集上宏平均 Pass/Ready/Exec. 为 40.6% / 87.4% / 91.9%，采集成功率 86.0%（165 Traj./h，1.17 Att./Succ.）。与 BODex 同协议（Dex3-1 + G1、520 对、每对 10 候选）对比：AnnotateAnything 达到 43.8% / 88.7% / 93.9%，高于 BODex 的 40.1% / 84.8% / 90.2%。
   - **真机零样本迁移（§5.5, H.7）：** 用仿真采集数据微调 $\pi_{0.5}$（每任务配置 1,000 条仿真演示，无真机微调），Franka 夹爪 + XHand 灵巧手完成 540 次单任务与 100 次长程（5 物体）评估：Grasp 93.0% (186/200)、DexGrasp 77.0% (154/200)、Insertion 70.0% (28/40)、Open Drawer 82.5% (33/40)、Close Lid 88.3% (53/60)、Long-horizon 61.0% (61/100)。注：无真机基线对照，仅作为 sim2real 可行性验证。
   - **局限（J.1）：** 依赖上游 VLM 与分割粒度（如酒杯未细分为杯身/杯柄/底座），严重遮挡/强柔性/精细接触仍较弱；仿真物理验证通过不代表真机无误；部分外部对比通过框架适配器转换，可能包含转换保真度误差。
