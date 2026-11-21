# UrbanVLA: A Vision-Language-Action Model for Urban Micromobility（arXiv:2510.23576）

> 来源归档（ingest · 原文通读）

- **标题：** UrbanVLA: A Vision-Language-Action Model for Urban Micromobility
- **类型：** paper
- **会议：** ICRA 2026
- **arXiv：** <https://arxiv.org/abs/2510.23576>（v2，2026-10-01）
- **PDF：** <https://arxiv.org/pdf/2510.23576>
- **项目页：** <https://pku-epic.github.io/UrbanVLA-Web/>
- **代码：** <https://github.com/GalaxyGeneralRobotics/UrbanVLA>
- **机构：** Peking University；Galbot；USTC；BAAI
- **作者：** Anqi Li*, Zhiyong Wang*, Jiazhao Zhang*, Minghan Li, Yunpeng Qi, Zhibo Chen, Zhizheng Zhang†, He Wang†
- **核查日期：** 2026-11-21
- **核对范围：** 通读 arXiv:2510.23576v2 全文正文（§I Introduction、§II Related Works、§III Method A–D 与 Fig. 2–3、§IV Experiments A–D 与 Tables I–III / Fig. 4、§V Conclusion）。
- **通读：** 已完成

## 核心技术与实验事实（原文核对）

- **任务与架构（§III-A/B）：** 面向城市微出行（delivery / assistive / guide robots）的路线条件 VLA：输入未来 $D=40$ m 按步长 $d=2$ m 采样的 20 个自车系坐标及转弯距离/方向“路书（roadbook）”文本指令 $I$ 与多视角滑窗 RGB 图像 $O_{vis}$，经 **DINOv2 + SigLIP** 拼接、网格池化与两层 MLP 投影接入 **Qwen2** LLM 主干（基于预训练导航基座 **NavFoM**）。双分支输出：VideoQA 走自回归语言头，导航取当前步 action token $E_{TA}$ 经 MLP ActionModel 解码 $N$ 步 $SE(2)$ 局部轨迹 $\tau$。
- **两阶段训练策略（§III-C）：**
  1. **SFT 阶段：** 在 MetaUrban PPO 专家仿真轨迹、Sekai 街景视频解析轨迹（MSE 监督）与 LongVU/Sekai VideoQA（交叉熵）上联合微调。提出 **Heuristic Trajectory Lifting（HTL）**——对原始轨迹去噪滤波、拐点分段后注入高斯位置扰动再平滑重采样——避免模型死记理想规划路线而在真机失真路书下失效。
  2. **RFT 阶段（IQL 离线强化学习）：** 在 MetaUrban 2,400 个仿真回合（约 40 小时）+ 北京真机遥操作 8 小时（刻意含负奖励次优样本以稳定 Q 学习）的混合数据上训练 $Q_\theta(s,a)$ 与 $V_\psi(s)$，状态编码取 SFT 模型 LLM **第 17 层隐状态** $H_T^{(17)}$，以 $r=\lambda_{comp}l_{completion}-\lambda_{coll}\mathbf{1}_{collision}-\lambda_{dev}\mathbf{1}_{deviation}$（系数 0.5 / 1 / 1）经 AWR 目标优化策略。
- **关键实验（§IV，MetaUrban-12K 与 Unitree Go2 真机）：**
  - **Table I（MetaUrban）：** 仅用 RGB 输入超越 7 个 LiDAR 基线。PointNav Test SR **94%** / SPL **0.91**、Unseen SR **97%** / SPL **0.95**；SocialNav Test SR **91%** / SNS **0.87**、Unseen SR **88%** / SNS **0.85**。
  - **Table II（HTL 消融）：** 去掉 HTL 在仿真 Test 略高（93% vs 89%），但在 60 m C 形人行道真机测试（路书折线偏差约 10 m）中路径完成度（RC）从 **100%** 暴跌至 **42%**。
  - **Table III（RFT 消融）：** RFT 后 SocialNav Unseen SR 从 82% 升至 **88%**，Cost 从 0.98 降至 **0.82**。
  - **真机部署：** Unitree Go2 四足挂载前/左/右/后 4 相机 + Jetson Orin + 5G CPE + GNSS + 高德（Amap）Web API 路书，连接远程 RTX 4090 服务器以 **2 Hz** 运行（配局部规划器执行轨迹），完成过街天桥、人行横道与 >500 m 连续城市街区导航（全程有安全员监控干预）。

## 对 wiki 的映射

- [UrbanVLA](../../wiki/entities/cn-os-urbanvla.md)
