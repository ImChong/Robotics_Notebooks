# OpenCap Monocular: 3D Human Kinematics and Musculoskeletal Dynamics from a Single Smartphone Video（arXiv:2603.24733）

> 来源归档（ingest）

- **标题：** OpenCap Monocular: 3D Human Kinematics and Musculoskeletal Dynamics from a Single Smartphone Video
- **类型：** paper / monocular video / biomechanics / motion capture / OpenSim / clinical
- **arXiv abs：** <https://arxiv.org/abs/2603.24733>
- **DOI：** <https://doi.org/10.48550/arXiv.2603.24733>
- **作者：** Selim Gilon*, Emily Y. Miller, Scott D. Uhlrich（University of Utah · Movement Bioengineering Lab, MoBL）
- **项目页：** <https://utahmobl.github.io/OpenCap-monocular-project-page/>
- **代码：** <https://github.com/utahmobl/opencap-monocular>
- **产品入口：** <https://opencap.ai>（iOS app + web app + 云端处理）
- **入库日期：** 2026-06-30
- **一句话说明：** 从 **单台静态智能手机视频** 估计 **3D 骨骼运动学** 与 **肌肉骨骼动力学**（关节力矩、GRF、肌力）；以 **WHAM** 初始化后经 **物理约束优化** 精炼，再经 **OpenSim IK** 与仿真/机器学习混合管线输出临床可解释指标。

## 摘要级要点

- **问题：** 传统实验室 marker mocap + 测力台成本高、耗时长，难以在诊所/居家大规模评估运动学与动力学（如坐站过渡中的股四头肌力、行走膝内收力矩）。
- **定位：** 在既有 **双相机 OpenCap**（低成本多手机）基础上，进一步把硬件门槛降到 **单台 iPhone/iPad + 三脚架**，<1 分钟完成采集，云端 <2 分钟出运动学。
- **管线五步：** (1) ViTPose + **WHAM** 估计 2D 关键点与 SMPL 全局姿态；(2) **两阶段 PyTorch 优化** 精炼相机外参与 SMPL（重投影、脚滑/穿地、关节速度惩罚）；(3) 从精炼 SMPL 网格提取 **虚拟皮肤 marker**；(4) **OpenSim 逆运动学** 得关节角；(5) **物理仿真 + ML** 估计动力学（行走 GRF 用 gait dynamics ML；其余活动见离线 post-processing 仓库）。
- **验证：** 对标 marker mocap + 测力台；行走/深蹲/坐站。旋转 DoF **MAE 4.8°**、骨盆平移 **3.4 cm**；相对纯 CV+IK 基线旋转精度 **+48%**、平移 **+69%**；行走 GRF 精度 **≥ 双相机 OpenCap**。
- **临床用例：** (1) 坐站膝伸展力矩区分衰弱前期（阈值 ~11 Nm）；(2) 行走膝内收力矩（OA 进展指标，阈值 ~0.5% BW·ht）。
- **部署：** HIPAA 合规 web app；PolyForm Noncommercial 1.0.0 许可；SMPL 模型仍受 MPI 非商业限制。

## 核心摘录（面向 wiki 编译）

### 1) 采集与产品形态

| 项 | 说明 |
|----|------|
| 硬件 | 单台 iPhone/iPad + 三脚架；**静态相机**（非手持跟拍） |
| 相机位姿 | 被试前方 **45°**；全身入画；<5 m |
| 已验证活动 | 行走、深蹲、坐站；**跳跃不支持** |
| 输出格式 | `mono.json`（OpenCap Visualizer）、`*.trc`/`*.mot`（OpenSim）、`*_scaled.osim` |

### 2) 优化目标（Stage 1 / 2 归纳）

- **Stage 1：** 固定 WHAM 姿态，优化体型 β 与相机外参 ξ；置信度加权 2D 重投影 + 身高先验 + β 正则。
- **Stage 2：** 联合优化 SMPL 姿态/平移/朝向；增加脚滑/穿地、关节速度、接触概率（WHAM heel/toe）等物理项。
- **假设：** 相机静止、体型时不变、身高由录制时查询、iOS 设备内参来自 2018 年后机型数据库。

### 3) 与相关系统对比

| 系统 | 相机数 | 输出 | 动力学 | 机器人重定向 |
|------|--------|------|--------|--------------|
| 实验室 Vicon + 测力台 | 多 + marker | 金标准运动学/动力学 | 是 | 需额外导出 |
| 双相机 OpenCap | ≥2 手机 | OpenSim 运动学 + 动力学 | 是 | 非主线 |
| **OpenCap Monocular** | **1 静态手机** | OpenSim 运动学 + 动力学 | **是** | TRC/MOT 可作参考，但面向临床生物力学 |
| GVHMR / WHAM 纯回归 | 1 | SMPL 世界轨迹 | 否 | 人形 GMR 上游更常见 |
| MAMMA | 多视角同步 | SMPL-X | 否 | 高精度双人 SMPL-X 采集 |

### 4) 代码与后处理仓库（README / SimTK）

| 仓库 | 作用 |
|------|------|
| [utahmobl/opencap-monocular](https://github.com/utahmobl/opencap-monocular) | 单目视频 → 运动学（WHAM + 优化 + OpenSim IK） |
| [opencap-org/opencap-processing](https://github.com/opencap-org/opencap-processing) | 运动学后处理 + 动力学仿真 |
| [opencap-org/opencap-processing-grf](https://github.com/opencap-org/opencap-processing-grf) | 行走 GRF 混合 ML–仿真 |

- 环境：Ubuntu 20.04/22.04、Python 3.9、NVIDIA driver ≥520；见 `installation/INSTALL_SLIM.md`
- 许可：PolyForm Noncommercial 1.0.0

## 对 wiki 的映射

- 沉淀实体页：[OpenCap Monocular](../../wiki/entities/paper-opencap-monocular.md)
- 项目页归档：[sources/sites/opencap-monocular-github-io.md](../sites/opencap-monocular-github-io.md)
- 代码归档：[sources/repos/opencap-monocular.md](../repos/opencap-monocular.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2603.24733>
- 项目页：<https://utahmobl.github.io/OpenCap-monocular-project-page/>
- 代码：<https://github.com/utahmobl/opencap-monocular>
- OpenCap 平台：<https://opencap.ai>
