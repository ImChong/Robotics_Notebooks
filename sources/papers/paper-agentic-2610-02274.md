# Awomo-SimDataEngine: Agentic Simulation-Ready World Generation

> 来源归档（paper）

- **标题：** Awomo-SimDataEngine: Agentic Simulation-Ready World Generation
- **作者：** Awomo-PhysicalRSI Team（核心贡献者含 Danjiao Ma、Enhui Ma、Haohan Liu、Heng Jia、Hui Shan、Jianhua Xu、Jiahuan Zhang、Jiangdi Xu、Kaiwen Guo、Kaicheng Yu、Linwei Zhang、Liyang Jin、Maochun Luo、Pengyao Niu、Shiwen Li、Shuangyu Feng、Tong Zhang、Tianheng Wang、Xin Wang、Xiangru Huang、Yongqiang Huang、Zhaozhi Wang、Zijian Ma）
- **机构：** Awomo-PhysicalRSI Team（项目页域名 `awomo-westlakedi.github.io`，论文未另列标准机构全称）
- **版本 / 载体：** arXiv:2610.02274v2 [cs.RO]（v1 提交于 2026-10-01，v2 修订于 2026-11-10；arXiv HTML 全文，CC BY 4.0）
- **原文：** [arXiv HTML v2](https://arxiv.org/html/2610.02274) / [arXiv 摘要页](https://arxiv.org/abs/2610.02274)
- **项目页 / 代码：** 项目页 <https://awomo-westlakedi.github.io/Awomo-SimDataEngine/>（正文标注但入库核查时 `project_page_status=0` 不可达；全文及检索未列官方 GitHub 仓库或可执行源码链接，开源状态判定：**宣称项目页但未列可运行开源代码链接 / 未确认开源代码**）
- **核对范围：** 已通读 arXiv HTML 全文（Abstract、§1 Introduction、§2 Related Work、§3 Overview、§4 Asset Generation / §4.1 ISArt / §4.2 Standardization、§5 Scene Generation / §5.1 Unravel / §5.2 SimForge、§6 Graph-Native Harness、§7 PolicyForge、§8 Experiments and Results、§9 Limitations and Conclusion，以及 Appendix A–E 全部公式与表 1–28）
- **通读：** 已完成

## 原文核对笔记

- **系统定位：** 面向机器人训练数据生成的可交互仿真世界构建与示范合成系统，将输入条件 $C=\{T_{\text{text}},I,Q\}$ 转化为经过验证的仿真世界与可重放（replayable）机器人示范。
- **资产服务与 ISArt（§4、Appendix B）：** 融合检索、图生 3D（Hunyuan3D、TRELLIS.2、SAM 3D）、程序化建模与面向单图铰接物体的 **ISArt**。ISArt 由四阶段构成：VLM 几何-运动学结构规划、基于 TripoSG 多部件 DiT 的结构条件部件生成（E 层级偏置 + C 结构交叉注意力，Rectified-Flow 损失训练 25,000 步）、分层运动学记忆引导的 URDF 绑定与 Isaac Sim 视频批判器校验。在 60 物体测试集上部件几何 CD 为 0.163（基线 TRELLIS.2+P3-SAM+X-Part 为 0.268）；在固定真值几何下绑定成功率为 100%（AxisErr 0.07、PivotErr 0.05、TypeErr 13.33%）。
- **双路径场景生成（§5、Appendix C）：**
  - **Unravel（图生场景）：** 无训练的规划-编辑-验证逐层拆解循环（内容物先于容器移除），利用未编辑锚点像素 $\mathcal{A}_k$ 对齐 MoGe3 点云并按射线最早存活轮次归属到物体簇 $C_i$，再自房间外壳逆序重组、双向中位 Chamfer 距离 + 截断相似性 ICP 配准（受欠观测轴下界约束），最后执行物理沉降与视觉审计。在 5 场景 160 物体上，盲评人工资产保真度由 4.9 升至 8.6（1–10 分）；在 20 场景照片相似度评测上，CLIP-I 0.854、LPIPS 0.632、Edge IoU 0.156。
  - **SimForge（文生场景）：** 面向单房间与多房间住宅，由语言模型规划拓扑、门窗与物体需求，再由几何感知例程落位并进入物理检查。在 179 条房间提示（SceneEval）与 31 条多房间提示（SceneSmith 协议）上相对 SceneSmith 数量一致性分别提升 6.5 与 6.4 个百分点、可达性提升 14.7 与 13.3 个百分点，检测碰撞率 0.0%、稳定性 100.0%。
- **Graph-Native Harness（§6、Appendix A）：** 将迭代生成原子化为 7 步显式状态转移（Observe→Diagnose→Attribute→Act→Validate→Select→Route），区分世界状态 $S=(R,O,A,X,H,J,P,M)$ 与证据状态 $E$，按表 5 因果归属将失败路由至最小责任模块并仅失效受依赖下游状态；只有通过语义、几何、物理、任务与导出合取门控 $\Phi(S,T,E)=1$ 才晋升为 Accepted World。
- **PolicyForge 与跨仿真器协同训练（§7–8、Appendix D–E）：** 在 Isaac Sim 中绑定 Franka、Piper 双臂、Agibot G1/G2 等载体，经任务分解、控制器路由（IK/cuRobo/RL/脚本）、重置采样与成功判据过滤导出 LeRobot 格式示范，并通过场景/任务 Data Cousins 构建 `SimData-Easy/Lite/Hard/Ultra`。在基于 MuJoCo 的 LIBERO-Plus（1,003 回合）上协同训练基于 Wan2.2-TI2V-5B 初始化的 WAM（32 GPU，1:1 有效帧比），13K 步预算内最佳 checkpoint 总成功率由 77.17%（Baseline 9K）提升至 89.43%（SimData-Ultra 13K，+12.26 pp），其中 Goal 泛化提升 31.66 pp、Spatial 提升 6.25 pp、Object 提升 8.34 pp，而长时域 LIBERO-10 仅提升 1.99 pp；在 LIBERO-Pro（1,000 回合）上最佳表列总成功率由 46.7% 升至 48.7%（13K）。
- **核心局限（§9.1）：** 单视角遮挡推理与图像补全存在尺度与不可见几何不确定性；未量化 Harness 修复效率与世界生成通过率；WAM 协同训练未做单模块受控消融与多随机种子置信区间，也未验证真机迁移；主要覆盖刚体与铰接体，未覆盖可变形体。

- 对 wiki 的映射：[Awomo-SimDataEngine 实体页](../../wiki/entities/paper-agentic-2610-02274.md)
