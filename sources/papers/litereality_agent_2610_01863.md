# LiteReality-Agent: An Agentic System for Interactable 3D Indoor Scene Reconstruction

> 来源归档（ingest）

- **标题：** LiteReality-Agent: An Agentic System for Interactable 3D Indoor Scene Reconstruction
- **类型：** paper / real2sim / indoor-scene-reconstruction / embodied-simulation
- **来源：** arXiv:2610.01863v1（cs.CV，2026-10-01）
- **作者：** Zhening Huang、Yueyan Li、Johnathan Chiu、Xiaoyang Lyu、Matt Zhou、Yuxin Yao、Joan Lasenby、Shangzhe Wu
- **机构：** 剑桥大学、帝国理工学院、独立研究者
- **论文：** <https://arxiv.org/abs/2610.01863> · <https://arxiv.org/html/2610.01863v1>
- **项目页与演示：** <https://litereality.github.io/agent/>
- **交互式 3D 查看器：** <https://litereality.github.io/agent/vr/index.html>
- **代码：** <https://github.com/LiteReality/LiteReality-Agent>（Apache-2.0）
- **配套扫描 App：** <https://apps.apple.com/gb/app/litereality/id6774158260>（App Store 免费）
- **入库日期：** 2026-10-15
- **最后更新：** 2026-10-15
- **一句话说明：** 将 iOS RGB-D 房间扫描编译为可编辑的 Room.py 场景程序，由 coding agent 通过 observe-edit-verify 闭环完成布局修复、物体重建和质量检查，并导出 MuJoCo / URDF / MJCF 环境。

## 核心论文摘录（MVP）

### 1) 把室内重建组织为可执行场景程序

系统以 LiteReality Scanner 采集的 RGB-D 帧、深度、相机标定、点云和 RoomPlan 布局为输入。确定性初始化阶段修复布局关系并重建物体；随后 coding agent 以 Room.py 为持久工作区，通过选帧、渲染对比、尺寸测量、墙面拼接和材质查询等工具迭代场景。

**对 wiki 的映射：** [LiteReality-Agent](../../wiki/entities/litereality-agent.md)

### 2) 用证据闭环驱动可编辑重建

Agent 每轮根据扫描观察修改场景程序。重建后重新编译并进行视觉对比、布局与几何检查；碰撞和支撑关系分别验证，未通过的检查会反馈给后续编辑。对象生成按结构需求路由至程序化建模或 TRELLIS.2，物理属性则保留人工定义值或按类别、尺度和几何估计。

**对 wiki 的映射：** [LiteReality-Agent](../../wiki/entities/litereality-agent.md)；[Articraft](../../wiki/entities/articraft.md)

### 3) 结果较强，但指标不等于物理真实性证明

论文在 10 个室内场景、620 个采集视角上评估。LiteReality-Agent 的 SSIM / PSNR / RMSE / LPIPS 均值均优于所列基线；几何表中深度 MAE 为 10.50 cm、86 个家具对象均满足底部近接触条件，且未检测到家具表面相交对。人工感知评分的布局、物体、整体相似度均值为 4.60、4.58、4.54 / 5。

这些结果覆盖视图一致性、局部几何与感知质量，不代表真实动力学验证。作者指出物理属性缺少 ground-truth 验证，部分几何指标不构成完整物理有效性证明；视觉评测也基于采集视角而非 held-out novel views。

**对 wiki 的映射：** [LiteReality-Agent](../../wiki/entities/litereality-agent.md)；[Sim2Real](../../wiki/concepts/sim2real.md)

## 开源与复现边界

- 项目页链接至公开 GitHub 仓库；仓库声明 Apache-2.0，README 提供 uv 安装、litereality run、stage author、stage simulate 和 view 入口。
- LiteReality Scanner 在 App Store 免费提供；example-scans 可用于没有自有扫描数据时试跑。
- 模型部署可走 Modal，也可使用 Linux 本地 GPU；官方 README 当前列出 Blender 5.x、图像生成 API、Agent CLI 等依赖。
- 仿真输出可用 MuJoCo，并提供 URDF / MJCF 导出；读者仍需按自己的机器人仿真任务校验坐标、碰撞体和估计物理参数。

## 当前提炼状态

- [x] 论文摘要、方法、评测与局限已核对
- [x] 项目页、交互查看器、App Store 与公开代码状态已核对
- [x] 映射到已有唯一实体页 [LiteReality-Agent](../../wiki/entities/litereality-agent.md)
