# G1 Football：NVIDIA ProtoMotions / Metatron 的 G1 视觉守门员浏览器演示

- **类型：** NVIDIA 官方交互演示与技术文档（site / demo）
- **英文标题：** G1 Football — ProtoMotions
- **中文标题：** G1 Football：基于视觉感知与 Metatron 的 Unitree G1 仿真守门员
- **项目演示：** https://nvlabs.github.io/ProtoMotions/demos/g1-football/
- **技术文档：** https://nvlabs.github.io/ProtoMotions/user_guide/g1_football.html
- **代码与发布仓库：** https://github.com/NVlabs/ProtoMotions/tree/main/demos/g1-football
- **仓库说明：** https://github.com/NVlabs/ProtoMotions/blob/main/demos/g1-football/README.md
- **基础策略项目：** https://research.nvidia.com/labs/par/projects/metatron/
- **模型权重：** https://huggingface.co/nvidia/Metatron-v1
- **机构：** 英伟达（NVIDIA / NVlabs）
- **核对日期：** 2026-11-12
- **核对范围：** 通读 G1 Football 官方用户指南的 Playing、Runtime and download、Known limits、Licensing 全部正文，以及公开的五点球页面说明；对照官方仓库 README 的控制、实验与源码开放边界。浏览器互动是否满足物理设备实时帧率未实测。
- **通读：** 已完成

## 核心事实与证据边界

1. 任务是浏览器内五次点球：用户通过 W/S 调整高度、A/D 调整左右、按住 Space 蓄力后松开射门；移动设备有触控操作。机器人是**仿真 Unitree G1**，而非实机控制器。
2. 物理由 MuJoCo WebAssembly 运行；机载相机图像与球检测在浏览器本地进行，守门员从相机推测球轨迹，**不使用隐藏的足球真值位置**。画面中的守门员相机 inset 是感知所用相机画面。
3. 使用 Metatron 策略：页面自动下载约 **403 MB** 的 FP32 权重（Hugging Face pinned revision），使用 SHA256 检查并尝试缓存。GPU 路径采用 ONNX Runtime Web / WebGPU，不能使用时可回退 CPU/WASM。真实运行速度取决于设备，官方**不保证始终实时**。
4. 远侧低/中高度球（由相机估计的球门横越位置距守门员侧方至少约 **1.2 m**）使用参考扑救动作，先侧移、下扑、保持低位直至球到达，最后交给原生恢复；来源为一个 BONES-SEED 动作片段。近侧球仍用原生定时扑救；高于约 **0.9 m** 的高角球不触发扑救下潜，因为缺少跳跃参考。
5. 官方受限对照：**24 次配对射门**（8 种射门 × 3 组相机噪声种子）中，加入下扑参考的版本救下 **21 次**，原生版本 **16 次**。这是受限测试，不是通用守门成功率；远侧慢球有退化，某些球仍会漏、甚至失衡。
6. 开放边界：官方仓库提供 **已编译浏览器应用资源包**、公开安装/检验脚本及模型权重；游戏与视觉检测的完整开发源码、source map **未公开**。因此不可称「完整开源守门员源码」。

## 软件/许可与部署注意

- 页面约需 403 MB 权重下载（冷启动还需 runtime 和场景资源）；清空缓存、存储配额不足可能重复下载。
- WebGPU 优先，`?provider=wasm` 可显式选择 CPU 路径；HTTPS 或 localhost 才能可靠使用隔离服务工作线程。
- 浏览器预览 CI 只验证静态部署、视觉运行时和预加载请求，**主动截断权重初始化**；未验证真机、iPhone/Safari 物理设备、真实 WebGPU 性能或完成游戏。
- ProtoMotions 著作的软件为 Apache-2.0；Metatron 权重为 NVIDIA Open Model License；游戏、感知依赖和机器人资产保留各自许可，需遵守资源包 NOTICE 与 licenses。
- 已核对的官方代码与部署步骤见 [G1 Football 发布仓库源码归档](../repos/g1-football-protomotions.md)。

## 对知识库的映射

- 独立演示实体：[G1 Football 视觉守门员](../../wiki/entities/g1-football-metatron-goalkeeper.md)。
- 基础策略复用：[Metatron 论文实体](../../wiki/entities/paper-metatron-promptable-whole-body-execution.md)、[ProtoMotions 框架](../../wiki/entities/protomotions.md)。
