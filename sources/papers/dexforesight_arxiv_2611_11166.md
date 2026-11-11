# DexForesight（arXiv:2611.11166）

> 来源归档（paper）

- **标题：** DexForesight: Grounding Dexterous Policies with Implicit Spatial Foresight
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.11166>
- **PDF：** <https://arxiv.org/pdf/2611.11166v1>
- **项目页：** https://dexforesight.github.io/
- **代码：** 论文与项目页静态内容均未给出代码仓库（未核实是否另有发布）
- **入库日期：** 2026-11-11
- **作者与机构：** Zhongxi Chen、Yi Yang 等，通讯作者 Zhijie Deng；Sharpa Robotics、上海交通大学（SJTU-Sharpa Joint Lab for Embodied AI）、香港中文大学、复旦大学（首页署名）
- **一句话说明：** 训练时用离线深度、机器人运动学手部点云与示范动作训练「动作条件空间世界模型」，再冻结它并把 query 特征对齐到策略 latent；部署时只用 RGB、语言、机器人状态。
- **核对范围：** 通过 arXiv v1 PDF 静态通读全文第 1–5 节、表 1–4、图 1–2 与参考文献（图 2 看了原图，其余图以图注为准）；v1 正文多处写「Appendix ??」「Figure ??」，PDF 内不含附录与补充材料，故附录里的核心设置表、真机可视化未能核对；项目页只静态抓取到脚本骨架，视频与交互内容未核实；无代码可读。
- **通读：** 已完成

## 开源状态

- 文章与论文均未给代码；项目页静态抓取未发现 GitHub 链接（未核实，2026-11-11）。

## 核心摘录

1. **方法（第 3 节）：** 世界模型主干初始化自 RynnBrain，深度经 OmniVGGT 编码后在第 2/6/10/14/18/22/26 层以交叉注意力注入，并有 token 级路由跨层聚合；世界 query 在第 12/20/28 层读隐状态；Hand Flow Expert 预测手部持久点位移，Visual Dynamics Expert 预测 DINOv3 特征残差（式 3）。
2. **对齐（第 3.2 节）：** 冻结世界模型，策略 query 通过 token 余弦损失与 pairwise 相似度关系损失对齐教师 query（式 5），再拼接进动作专家上下文；教师 query 由示范动作 U_t 条件化，策略不观测 U_t。
3. **数据（第 3.1 节、图 3）：** 约 370 小时示范、60 任务、五类；其中 100 小时子集用 StereoAnyVideo 离线估计深度（空间摆放、装配接触、旋转关节三类）。
4. **仿真（表 1，第 4.2 节）：** 11 个 DexJoCo 任务，每任务 100 次；DexForesight（GR00T N1.7 基座）总成功率 60.5%，GR00T N1.7 为 49.4%，Fast-WAM 50.9%，Cosmos3 49.5%，π0.5 35.1%。
5. **真机（表 2，第 4.1–4.2 节）：** Sharpa North 机器人，22 自由度 Sharpa Wave 手；3 个任务各 20 次、held-out 物体位置；DexForesight 58.3%，WLA 50.0%，GR00T N1.7 48.3%，QwenVLA 30.0%，Cosmos3 30.0%，Fast-WAM 10.0%。
6. **推理开销（第 4.2 节）：** QwenVLA 227.4 ms，DexForesight 230.4 ms（+3.0 ms，1.3%）；Cosmos3 生成 16 帧未来视频约 930 ms（仅视频生成阶段）。
7. **探针与消融（表 3、表 4，第 4.3–4.4 节）：** 冻结 latent 的线性岭回归探针预测 0.54 s 后手部点位移，误差从 13.59 mm 降至 10.01 mm；去掉离线深度，仿真 49.7%、真机 43.3%。

## 对 wiki 的映射

- [paper-dexforesight](../../wiki/entities/paper-dexforesight.md)
