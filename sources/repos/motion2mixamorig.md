# Motion2MixamoRig

> 来源归档

- **类型：** repo / motion-retargeting
- **链接：** https://github.com/MaxLiu22/Motion2MixamoRig
- **入库日期：** 2026-10-13
- **一句话说明：** 单人视频或静态照片经人体动作恢复后，重定向到 Mixamo 角色，并导出带蒙皮 GLB 与多视角预览。
- **沉淀到 wiki：** 是

## 核心内容

- 仓库公开 v0.1.0 CLI 与 Blender 4.2+ 扩展；命令行和 Blender 插件调用同一条推理流程。
- README 描述推理依赖 GVHMR 等模型权重；项目不是新的 HMR 论文，定位是把单人输入动作转换为可复用角色骨架动画。
- 常用入口：m2mr doctor 检查环境，m2mr run 运行；支持 --video、--image、--rig、--skeleton。主要产物包括 skeleton_motion.npz、mixamo_rotations.npz、mixamo_character.glb 和预览视频。
- 初次推理会下载约 5 GB 权重；复用已提取 skeleton 并更换角色可缩短后续转换时间。README 给出 CPU 视频处理耗时估计，实际受长度与硬件影响。
- 输入素材需自行准备：单人全身视频/照片、Mixamo FBX、SMPL-X body model。SMPL-X 默认许可限制非商业使用；Mixamo、GVHMR及其他权重分别遵循上游条款，不能因为本仓代码 Apache-2.0 就推断全部素材/权重可自由再分发。

## 链路

~~~mermaid
flowchart LR
  A[单人视频 / 照片] --> B[人体动作恢复<br/>GVHMR 等权重]
  B --> C[骨架运动序列]
  C --> D[Mixamo 骨架重定向]
  E[Mixamo FBX / 目标角色] --> D
  D --> F[蒙皮 GLB + 预览]
~~~

## 来源

- 仓库 README：<https://github.com/MaxLiu22/Motion2MixamoRig>
- 上游 GVHMR：<https://github.com/zju3dv/GVHMR>；论文 <https://arxiv.org/abs/2409.06662>

