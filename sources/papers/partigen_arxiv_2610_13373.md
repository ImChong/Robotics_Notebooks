# PartiGen 论文来源归档

## 元数据

- **题目：** PartiGen: Part-Aware Representations for Language-Conditioned Humanoid Motion Generation
- **作者：** Chuxuan Huang, Zhanxiang Cao, Tianyue Zhan, Haoyang Li, Yang Zhang, Liyun Yan, Chenlin Ming, Keqi Zhu, Yizhi Chen, Shengcheng Fu, Zihao Zhou, Xinyi Peng, Kaixuan Zhang, Boyang Xing, Jiawei Bao, Yang Yang, Yue Gao
- **机构（arXiv HTML）：** 复旦大学、上海创智学院、上海交通大学、浙江大学、同济大学、香港中文大学、Humanoid Robot (Shanghai) Co., Ltd.、北京航空航天大学。投稿信息未列出国家地方共建人形机器人创新中心；应以论文作者署名为准。
- **预印本：** https://arxiv.org/abs/2610.13373
- **HTML 全文：** https://arxiv.org/html/2610.13373v1
- **版本：** v1，2026-10-09，cs.RO
- **项目页 / 代码：** 截至 2026-10-14，arXiv 论文未提供独立项目页或官方代码仓库链接；未找到可核实的官方实现。
- **读取方式：** arXiv HTML 全文，2026-10-14

## 核心论文摘录

### 1) PartiGen: Part-Aware Representations for Language-Conditioned Humanoid Motion Generation（Huang et al., 2026）
- **链接：** <https://arxiv.org/abs/2610.13373>
- **核心贡献：** 提出部位感知运动 VAE（BP-MVAE）与时长自适应潜扩散（DA-LDM）：在共享全身表征中按解剖部位选择关节特征并融合局部/全身信息；通过正向运动学监督提升运动学一致性；在生成候选序列后预测逐帧 EOS，确定动作终点。人体 HumanML3D 重定向至 Unitree G1 的 segment FID 为 3.165，较 TextOp 降低 16.8%；时长 MAE 从固定截断的 12.471 帧降至 3.080 帧。冻结 GRIT 跟踪器在 MuJoCo 的 500 条参考中有 409 条通过姿态稳定筛查；经仿真筛查和人工挑选的 27 次真机尝试中，作者判定 26 次成功。
- **对 wiki 的映射：**
  - [PartiGen 论文实体页](../../wiki/entities/paper-partigen.md)
  - [基于扩散模型的运动生成](../../wiki/methods/diffusion-motion-generation.md)

## 阅读边界

- 真机结果对应经过仿真筛查与人工选择的少量动作，不能视为未经筛选的总体成功率。
- 生成器与 GRIT 执行控制器分开训练；论文评估的是生成参考以及冻结控制器的跟踪，不是联合训练的生成—控制策略。
- 因论文未链接官方项目页或可运行源码，仓库运行流程与训练复现细节无法独立验证。
