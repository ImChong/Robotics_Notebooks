# DiffWAM: A Fast and Efficient Navigation World Action Model

> 来源归档；入库及核查日期：2026-10-12

- **论文：** <https://arxiv.org/abs/2609.39763>
- **PDF：** <https://arxiv.org/pdf/2609.39763>
- **全文：** <https://arxiv.org/html/2609.39763v1>
- **首次提交：** 2026-09-30
- **类型：** paper / UAV navigation / predictive representation / trajectory distillation
- **作者：** Mo Zhu、Yuze Wu、Xijie Huang、Xiao Cui、Fei Gao、Xin Zhou
- **机构：** 浙江大学（Zhejiang University）、Differential Robotics
- **官方项目页：** <https://zzmmzzm.github.io/diffwam.github.io/> — [核查归档](../sites/diffwam.md)
- **开放核查：** 未发现 DiffWAM 专属可运行实现、权重或 DiffWAM-1000 发布入口；项目页指向 NavGen/FlyMirage 配套数据项目，不能视为完整复现包。
- **一句话说明：** 将视频模型早期预测特征与首帧几何接地为相机轨迹，以同一次视频 rollout 的离线重建监督替代部署时完整生成和重建。

## 核心摘录及知识映射

### 1. 输入与计算轴（§3.1–3.3）

RGB 与指令输入冻结 MiniMax H3；标准配置保留去噪 schedule 0–3 四次 backbone evaluation，Flash 仅保留第 0 次；最后所需 evaluation 在零基 DiT block 35 后截断。读取 15/25/35 三个深度，Grid-Motion 保留空间关联后才池化，Latent2Pose 用首帧几何生成 38 个未来姿态，加初始姿态共 39，名义约五秒。

**对 wiki 的映射：** [DiffWAM 主节点](../../wiki/entities/paper-diffwam.md) — 区分 schedule、网络层和物理时间，不把单次 Flash 推理推广为标准配置。

### 2. 离线教师与冻结边界（§3.4）

S0 骨干准备、S1 导航读出预训练、S2 GPD。教师继续完成学生所对应的同一 rollout，用 π³ 重建及 MoGe2 深度标定平移尺度；学生只见早期特征和首帧几何。S1/S2 冻结视频/几何骨干，训练读出组件；教师是估计米制重建参考，不是独立实测物理真值。

**对 wiki 的映射：** [DiffWAM 主节点](../../wiki/entities/paper-diffwam.md)、[Fast-WAM](../../wiki/entities/paper-fast-wam.md) — 直接预测特征读出与部署免未来像素解码的不同实现。

### 3. 异步执行（§4）

FastDreamer 复用条件路径的视觉语言参数改写局部指令，并行预测与首帧几何，在已提交轨迹执行期间准备下一候选。保留 capture/launch/ready/handoff/expiry 时间和任务进度；按采集时刻位姿转换世界坐标，计划交接前复核最新状态与可行性，不按推理耗时简单删除候选前缀。

**对 wiki 的映射：** [DiffWAM 主节点](../../wiki/entities/paper-diffwam.md) — 模型候选、下游规划、有效轨迹交接与任务管理的分工。

### 4. 评测和延迟（Tables 2–6、§5.3）

DiffWAM-1000 端点 SR 74.40%，IndoorUAV-VLA 56.77%，UAV-FLOW-Sim 91.42%；端点阈值室内 1 m、室外 3 m，不等于穿越/绕行完整闭环成功。架构对比给 RMSE 0.3492 m，但该对比不是所有 benchmark 配置的同一测量。Thor 的 Flash P50/P95 为 1.080/1.101 s、峰值 97.60 GiB；表格为 archived FastDreamer head/pipeline，排除指令改写、外部通信、文件输出和下游规划验收，不能拼成当前解码器整机吞吐。

**对 wiki 的映射：** [DiffWAM 主节点](../../wiki/entities/paper-diffwam.md) — 保留协议、配置与证据范围。

## 来源差异与保留边界

项目页把标准方法多处写成一次 truncated forward，而论文 §3 区分标准四次和 Flash 一次；项目页 mixed-layer FDE 写 0.3151，论文 Tables 4/5 为 0.4357，0.3151 是 ADE。项目页还混用历史 development set 结论。当前 wiki 以 arXiv v1 的具体章节/表格为准，保留差异，不将不同设置拼成一个性能承诺。
