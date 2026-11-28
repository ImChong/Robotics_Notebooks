# MAWAM 论文一手来源归档

- **英文标题：** MAWAM: A Multi-Agent World-Action Model for Coordinated Manipulation
- **中文标题：** MAWAM：用于协同操作的多智能体世界–动作模型
- **机构：** NVIDIA（英伟达）、The University of Hong Kong（香港大学）、MIT（麻省理工学院）
- **作者：** Xianzhe Fan, Chi-Pin Huang, Fu-En Yang, Jing Wang, Yecheng Wu, Yu-Cheng Chou, Johan Bjorck, Linxi “Jim” Fan, Jan Kautz, Yu-Chiang Frank Wang, Zhiding Yu, Hengshuang Zhao, Zhiqi Li
- **原文：** https://arxiv.org/abs/2611.25105v2
- **全文 HTML：** https://arxiv.org/html/2611.25105v2
- **PDF：** https://arxiv.org/pdf/2611.25105v2
- **项目页（论文所列）：** https://research.nvidia.com/labs/lpr/mawam/
- **版本与日期：** 2026-11-18 首次提交；2026-11-20 修订至 arXiv v2；2026-11-29 复核
- **核对范围：** v2 PDF 与 HTML 正文 §§1–6、Figure 1–4、Tables 1–3、Appendices A–E（数学与归属、控制接口、训练超参、评测与消融、真机平台和推理时延）。
- **通读：** 已完成

## 方法与证据摘录

1. 从预训练单机器人 GR00T 2 World-Action Model 初始化。每机器人私有相机流、状态和动作 token 拥有独立归属，通过共享 Transformer 内 dense cross-agent attention 联合去噪，分别以 future-view / action flow-matching loss 训练（§3）。
2. 每个联合样本预测各机器人 24 个控制步及每个私有视角 8 个未来帧；执行前 8 步后同步观测重规划。sim 20 Hz、real 15 Hz；共享全局相机只用于仿真条件，不是监督的未来目标（§3 / Appendix B）。
3. View–agent map 保留相机归属；regular-simplex RoPE identity、action-to-video bias、独立的 task-role embedding 区分“是谁”和“负责什么”。身份随机分配和整机器人组打乱鼓励近似排列等变，但不能当严格等变（§3.3 / Appendix A）。
4. 训练分单机器人 Warm-up 和同步多机器人 Team Training；去 Warm-up 四臂 Long Pipeline Delivery 从 95% 降至 34%（§3.4 / Table 3）。
5. **真机（Table 1）**：两台 Galaxea R1 Pro，每台 3 路私有相机，不使用共享相机；洗衣袋收纳 53.3%、浴巾折叠 63.3%、雨伞交接 86.7%，三任务各 30 次、均值 **67.8%**。相同示范/平台的 MA-VLA（π0.5 重实现）均值 **48.9%**，差 **18.9 个百分点**。
6. **仿真（Table 2）**：RoboFactory 8 任务、2–4 条 Panda 机械臂；每任务 150 条团队同步示范、100 回合，MAWAM **90.3%**，论文引用 MA-VLA 已发表平均 **83.4%**。四臂 Long Pipeline Delivery 为 95%，Take Photo 为 97%，三臂 Stack Cube 仅 59%。
7. **消融（Table 3 / Appendix D）**：四臂交接移除跨智能体注意力 95%→39%；Place Food 移除团队未来视频监督 78%→47%，去 task-role 78%→0%；将全队当单一集中控制器四臂交接为 0%。
8. **推理时的辨析**：测试时冻结未来 token 去噪，很多任务保持接近基线，但彻底移除未来 token，四臂交接 95%→0%；因此“训练未来监督”“保留未来 token”“部署是否生成完整未来画面”并不等价（§4.3 / Table 8）。
9. **部署/时延（Appendix E）**：真机单 H100、2 步联合去噪中位模型时间 **0.32 s**，执行 8 步约 **0.53 s**，不包含网络/物理执行。仿真单 L40 四步去噪四臂单样本 **1.85 s**，不能按真机时延外推。

10. **协作时序与视频度量**：16 步队友暂停令接收者的启动延迟中位数为 +12 步（独立策略 +6），干预下成功率 99% 对 44%；变化画面上未来预测 PSNR 比复制末帧高 4–7 dB，而整帧分数不占优。参见图 4、附录 C.2 与 D。

## 局限

- 作者实验只验证固定机器人组（最多 4 个仿真代理/机械臂），不涵盖成员动态增减、异构传感器/本体或人机共同操作；真机测试范围只含 3 任务 × 30 次。
- 正式 NVIDIA 项目页已可读取，代码入口仍为待发布；[项目页核查](../sites/mawam-nvidia-project.md) 区分演示与可下载工件。

## 对 wiki 的映射

- [MAWAM 唯一实体](../../wiki/entities/paper-mawam-coordinated-manipulation.md)
- [World Action Models](../../wiki/concepts/world-action-models.md)
- [Joint WAM 纵深](../../roadmap/depth-wam.md)
