# mjlab 技术报告通读（架构与 manager-based API 视角）

> 来源归档（原文通读 · 支撑 `wiki/entities/mjlab.md`）

- **标题：** mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning
- **作者：** Kevin Zakka, Qiayuan Liao, Brent Yi, Louis Le Lay, Koushil Sreenath, Pieter Abbeel（UC Berkeley / Sorbonne University）
- **类型：** paper / framework / technical-report
- **原文：** <https://arxiv.org/abs/2601.22074>（2026-11 通过 arXiv PDF v2 获取全文，pdftotext 通读）
- **一句话说明：** 技术报告，把 Isaac Lab 的 manager-based 环境 API 与 MuJoCo Warp 的 GPU 物理结合，强调单命令安装、MuJoCo 原生数据结构可见、PyTorch 原生接口。
- **核对范围：** 通读全文 §1–§9（摘要、引言、设计哲学、架构、组件、manager-based API、三个参考任务、软件设计、采用情况）及参考文献列表（11 页）。该报告不含定量基准实验（无吞吐 / 对比数表），因此无实验数字可核对；未阅读其引用的 Isaac Lab、MuJoCo Warp、BeyondMimic 等原文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位（§1–§2）：** Isaac Lab 需 Omniverse 运行时、安装与启动重；MuJoCo Playground 单体环境易改但多机器人 / 多任务下代码重复。mjlab 三项承诺：安装摩擦最小、物理透明可检视、与 MuJoCo 生态紧耦合；刻意只支持 MuJoCo Warp 单一后端（跨模拟器可移植为非目标），不做照片级渲染，视觉策略建议先训特权策略再蒸馏。
- **架构（§3，图 2）：** 实体（MJCF）合成为 MjSpec，在 CPU 编译为 MjModel，经 put_model / put_data 送上 GPU；MjData 带前导 world 维，一个对象保存 N 个并行世界；模型参数默认共享，需随机化时按字段扩成 per-world；仿真步以 CUDA graph 捕获并重放以消除 CPU 派发开销。
- **组件（§4）：** 单一 Entity 类（固定 / 浮动基座 × 有 / 无关节）取代 Isaac Lab 的多个类；传感器分层（MJCF 原生 + BVH 光线投射、接触、GPU 相机）；执行器分层（MJCF 原生 / MuJoCo 原生类型 / GPU 上的理想 PD、DC 电机、学习式 MLP 执行器，另有量化到物理步长的延迟包装）；地形含基元与高度场两族、带难度参数、支持课程网格；查看器为原生 MuJoCo 与基于 Viser 的网页版。
- **step 流水线（§5）：** Act → Simulate（decimation 个子步）→ Terminate → Reward → Reset → Command → Apply Events → Observe。Manager 包括 action / termination / reward / curriculum / event / command / observation / metrics；奖励按控制步长自动缩放；终止管理器检测 NaN/Inf 并保留滚动状态缓冲供回放；域随机化覆盖 body、joint、geom、camera、light、material、tendon，惯量随机化采用 Rucker & Wensing 的伪惯量参数化以保持物理一致；观测管线支持裁剪、缩放、噪声、延迟、历史，并支持 policy / critic 多组（非对称 actor-critic）。
- **任务（§6）：** 速度跟踪（G1 / Go1）、动作模仿（G1，DeepMimic + BeyondMimic 扩展）、操作（YAM 机械臂抬方块）；三种机器人形态取自 MuJoCo Menagerie。
- **软件设计（§7）：** TorchArray 零拷贝把 Warp 数组暴露为 PyTorch 张量；实例式配置（typed dict，非继承 dataclass）；tyro 自动生成 CLI；配置与实现同文件；pyright + ty 静态类型检查、40+ 测试文件；训练用 RSL-RL，多卡用 torchrunx；`uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096 ...`。
- **采用（§8）：** 用于 UC Berkeley 研究生机器人课程（ME 292b/193b）。

## 对 wiki 的映射

- [`wiki/entities/mjlab.md`](../../wiki/entities/mjlab.md)
