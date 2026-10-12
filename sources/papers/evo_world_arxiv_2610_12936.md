# Evo-World 论文来源归档

## 元数据

- **题目：** Evo-World: A Lightweight Vision-Language-Action Model with Future Motion Prediction via Optical-Flow Supervision
- **作者：** Zile Zhong, Yuxin Du, Tao Lin, Haoxiao Wang, Wei Dai, Binhua Zuo, Bo Zhao（通讯：Bo Zhao）
- **单位（HTML 作者栏）：** 上海交通大学人工智能学院（SJTU）；浙江大学；电子科技大学；QuicRobot
- **预印本：** https://arxiv.org/abs/2610.12936
- **HTML：** https://arxiv.org/html/2610.12936v1
- **版本：** v1，2026-10-09，cs.RO，CC BY 4.0
- **代码：** https://github.com/MINT-SJTU/Evo-World（2026-10-12 实测可克隆，见 [evo-world.md](../repos/evo-world.md)）
- **读取方式：** arXiv 摘要页 + HTML 全文；2026-10-12 抽取

## 要解决的问题

轻量 VLA 只用动作监督训练时，没有任何信号鼓励模型理解“场景接下来怎么动”。直接预测未来 RGB 图像又会把大量静态外观混入目标，浪费容量。Evo-World 把预测目标换成光流渲染图，使预测信号只关注运动；并通过表征多样性正则与停止梯度，防止预测表征在联合训练中被动作损失破坏。

## 方法与训练信息

- **骨干：** InternVL3-1B（InternViT 视觉编码 + InternVL3 LLM），全模型 0.87B 参数（含运动分支；原文未单列运动分支参数量）。
- **动作头：** flow-matching Transformer 动作专家，action chunk 长度 50；噪声动作 token 对 FiLM 调制后的 VL token 与状态 token 做 cross-attention；推理用 N 步前向 Euler。
- **运动分支：** DINOv2 初始化的图像编码器 E_I 编码指定的一个视角得到 z_t；CLIP 初始化的文本编码器编码指令；MLP 预测器输出 z_pred；监督目标 z_h 是“当前帧到未来第 15 帧的 RAFT 光流渲染图”经同一 E_I 编码的结果（仅训练时使用）。光流渲染使用按数据集固定的幅值尺度（取前 5 条 episode 幅值 99 分位标定）。
- **条件注入：** (γ,β)=α·tanh(G([sg(z_t); sg(z_pred)]))，α=0.1，最后一层零初始化使调制起点为恒等；以 FiLM 调制 VL token。
- **损失：** L = L_action + λ_world(L_pred + λ_sig·L_SIGReg)，λ_world=0.1，λ_sig=0.09；L_pred 为 z_pred 与 z_h 的 MSE；SIGReg 来自 LeJEPA，用 1024 个随机方向，对 z_t 与 z_h 施加。动作损失梯度经 stop-gradient 不进入运动模块。
- **两阶段训练：** 阶段 1 共 10,000 步冻结骨干，训练运动模块、FiLM 与动作专家；阶段 2 共 100,000 步解冻骨干端到端。
- **超参与硬件：** AdamW，峰值学习率 1e-5，权重衰减 1e-3，warmup 1,000 步 + cosine，梯度裁剪 1.0；全局 batch 64；动作专家 dropout 0.2；策略图像 448×448、运动分支 224×224；BF16 + DeepSpeed ZeRO-2；训练 4 张 RTX 4090；真机推理用 NVIDIA H20。无机器人数据预训练。
- **数据：** Meta-World MT50（Easy/Medium/Hard/Very Hard 四档）、RoboTwin 2.0（Clean/Random）；真机 AgileX 双臂每任务 100 条遥操作示范、共 300 条。仿真示范数量原文未报告。

## 论文报告的结果

- **Meta-World MT50（成功率 %，分档平均）：** Evo-World 91.1（Easy 93.2 / Medium 89.1 / Hard 88.3 / Very Hard 94.0）；Evo-1 80.6；FabriVLA 90.0；LA4VLA 87.5；Evo-Depth 84.4；π0 47.9。13.0% 相对提升即 91.1/80.6。Very Hard 档 LA4VLA 的 100.0 高于 Evo-World。
- **RoboTwin 2.0：** Evo-World Clean 67.1 / Random 16.5 / 平均 41.8；Evo-1 为 65.7 / 12.5 / 39.1；Xiaomi Robotics-0 平均 40.5（Random 18.2 高于 Evo-World）；π0 平均 31.4。
- **AgileX 真机（每任务 20 次）：** Floral arrangement 55.0、Mug hanging 70.0、Microwave placement 65.0，平均 63.3；π0 平均 51.7，OpenVLA-OFT 平均 23.3。资源：Evo-World 0.87B / 2.54 GB 显存 / 6.16 Hz；π0 3.50B / 6.69 GB / 5.42 Hz；OpenVLA-OFT 7.73B / 15.56 GB / 2.41 Hz。
- **消融（Meta-World 分档平均）：** 无辅助预测 80.6；预测未来 RGB 84.3；预测光流 91.1。融合方式：拼接 53.2 / cross-attention 86.4 / FiLM 91.1。去掉 SIGReg 85.8（-5.3）；去掉梯度分离 82.2（-8.9，Hard 档降到 68.3）。FiLM 仅 z_t 83.7、仅 z_pred 85.5、z_t+z_pred 91.1。
- **线性解码（附录 D）：** 光流监督下 z_t 的测试 R² 0.661、z_pred 为 0.508；论文自己指出这并不证明预测器比 z_t 多提供可解码的运动信息。

## 复现与解释边界

- 运动目标是 2D 图像平面位移，无显式 3D 几何；运动分支只用一个指定视角；光流需离线预计算且间隔固定为 15。
- 仿真基线数字大量引自其他报告，预训练与数据设定不一致；消融与主表未报告随机种子方差；真机每任务仅 20 次。
- 效率数字依赖所测部署配置。
- 本归档读取了 HTML 正文与附录要点；RAFT 具体权重名原文未写。
