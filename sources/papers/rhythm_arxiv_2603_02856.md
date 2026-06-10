# Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids

> 来源归档（ingest）

- **标题：** Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids
- **类型：** paper
- **来源：** arXiv preprint
- **原始链接：**
  - <https://arxiv.org/abs/2603.02856>
  - 项目页：<https://hoshi-no-ai.github.io/Rhythm/>
- **机构：** 作者单位含上海交通大学、香港中文大学（深圳）等（见 arXiv 作者脚注）
- **入库日期：** 2026-06-10
- **一句话说明：** 首个在 **真机双 Unitree G1** 上实现 **物理耦合全身交互**（拥抱、共舞、问候等）的统一框架：**IAMR** 解耦自运动与交互几何生成可行双人参考，**IGRL** 以图结构奖励学耦合动力学，**相对定位 + 相位同步** 桥接仿真全局可观与真机 ego-centric 部署；并发布 **MAGIC** 双人交互数据集。

## 核心论文摘录（MVP）

### 1) 问题：单人敏捷 ≠ 双机物理耦合交互

- **链接：** <https://arxiv.org/abs/2603.02856> §I
- **摘录要点：** 单人 locomotion / WBT 已较成熟，但 **多 humanoid 物理协作** 仍稀缺：图形学多体交互偏视觉保真、机器人侧多为人–机或被动物体；**异构人体→同构双机** 的重定向存在 **kinematic conflict**（个体流形 vs 统一交互流形不可兼得），且现有 tracking 策略常把 agent **孤立** 建模，无法刻画紧密接触下的 **耦合动力学**；仿真全局可观与真机 **异步、局部可观** 进一步放大 sim2real 鸿沟。
- **对 wiki 的映射：**
  - [Rhythm](../../wiki/entities/paper-rhythm-dual-humanoid-interaction.md) — 问题定义、与 AssistMimic / 单人 GMT 的分野

### 2) IAMR：拓扑分区 + 解耦能量优化

- **链接：** <https://arxiv.org/abs/2603.02856> §III-A
- **摘录要点：**
  - 以 **Interaction Mesh** 建双体图 $\mathcal{G}=(\mathcal{V},\mathcal{E})$，边集拆为 **intra-agent** $\mathcal{E}_{self}$ 与 **inter-agent** $\mathcal{E}_{inter}$。
  - **Kinematic conflict**：$\mathcal{M}_{ind}$（逐人缩放保自运动）破坏相对几何 →「空手握手」；$\mathcal{M}_{uni}$（全局统一缩放）保交互但致脚浮等形态冲突。
  - **解耦目标**：$E_{self}$ 对齐 $\mathcal{M}_{ind}$ 的 Laplacian + 关键连杆旋转；$E_{inter}$ 以距离衰减刚度 $\omega_{ij}=\omega_{max} e^{-\gamma d_{ij}}$ 约束跨体边对齐 $\mathcal{M}_{uni}$。
  - 额外导出 **Interaction Graph** 与 **Contact Graph** 作为下游 IGRL 的拓扑先验。
- **对 wiki 的映射：**
  - [Rhythm](../../wiki/entities/paper-rhythm-dual-humanoid-interaction.md) — IAMR 机制、与 GMR / OmniRetarget / DOR 基线对比

### 3) IGRL：MAPPO + 图奖励 + 课程采样

- **链接：** <https://arxiv.org/abs/2603.02856> §III-B
- **摘录要点：**
  - **MA-MDP + CTDE**，部署用 **MAPPO**；观测 $o_t=\{o_{prop}, o_{peer}, o_{ref}\}$，其中 $o_{peer}$ 含伙伴关节与 ego 系相对 root 位姿。
  - **图奖励**：$r_{inter}$ 惩罚交互边距离偏差（继承 IAMR 的 $\omega_{ij}$）；$r_{contact}$ 对齐接触图并正则化接触力，抑制穿透与非接触阶段的虚假力。
  - **训练策略**：误差感知 **adaptive RSI 课程**（失败 / tracking / 交互多目标）；**双体域随机化**（延迟噪声 peer 观测、初始位姿扰动）。
- **对 wiki 的映射：**
  - [Rhythm](../../wiki/entities/paper-rhythm-dual-humanoid-interaction.md) — Mermaid 流程、奖励分解与 ablation

### 4) 真机部署、MAGIC 数据集与主要结果

- **链接：** <https://arxiv.org/abs/2603.02856> §III-C, §IV
- **摘录要点：**
  - **部署**：POINT-LIO + GICP 地图配准 + Kalman 融合；LCM 广播全局位姿，ego 重建 $o_{peer}$；**相位 $\phi$ 软同步** $\dot{\phi}_{ego}=1+k(\phi_{peer}-\phi_{ego})$ 抗时钟漂移。
  - **MAGIC**：约 **3 小时** 光学 MoCap（身高匹配演员、>10 s 长序列），五类语义（协调 / 亲密护理 / 接触 / 社交仪式 / 竞争）；将发布 **原始 BVH + IAMR 重定向参考**。
  - **重定向 Q1**：IAMR 在 Intensive Contact 上 **IPR=0**、Contact F1 优于 GMR/OR/DOR；Inter-X 上 DSR **69.9%** vs DOR **52.9%**。
  - **策略 Q2**：去 peer obs / interaction rew / contact rew 均显著降成功率；**真机 G1** 演示拥抱、共舞、问候、肩并肩及扰动鲁棒性（项目页视频）。
- **对 wiki 的映射：**
  - [Rhythm](../../wiki/entities/paper-rhythm-dual-humanoid-interaction.md) — 数据集、指标、真机局限与推荐继续阅读

## 引用（项目页 BibTeX）

```bibtex
@article{chen2026rhythm,
  title={Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids},
  author={Hongjin Chen and Wei Zhang and Pengfei Li and Shihao Ma and Ke Ma
          and Yujie Jin and Zijun Xu and Xiaohui Wang and Yupeng Zheng
          and Zining Wang and Jieru Zhao and Yilun Chen and Wenchao Ding},
  journal={arXiv preprint arXiv:2603.02856},
  year={2026},
  url={https://arxiv.org/abs/2603.02856}
}
```
