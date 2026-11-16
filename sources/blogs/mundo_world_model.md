# Mundo: From seeing a place to simulating what happens there

> 来源归档（作者官方技术博客，2026-10-23）

- **原文：** <https://huggingface.co/blog/Luo-Yihang/mundo-world-model>
- **发布日期：** 2026-10-23；**复核日期：** 2026-11-13
- **核对范围：** 通读官方博客 TL;DR、How Mundo works（代理建立、代码世界、代理帧与渲染、训练）、应用、局限、参考文献与 Citation；并交叉检查项目页的技术演示描述。未对视频帧作独立定量测量。
- **通读：** 已完成
- **项目主页与开放状态：** [项目页归档](../sites/mundo-world-model.md)
- **仓库：** [官方仓库归档](../repos/mundo.md)

## 原文要点（归纳，非原文转录）

1. **代理几何：** 3D 模型恢复位姿与密集几何、检测器初始化 bounding boxes；编辑 VLM 通过合成重投影对照与多视图边缘数值拟合修正，reviewer agent 跨全视图审阅。
2. **物理可编程：** 场景程序保存物体、位姿与物理参数，通过 MuJoCo 等引擎推进动作、接触与重力状态；示例 Python API 是作者明确标注的简化表达。
3. **视觉条件：** Proxy Frames 按类别平涂，传递仿真运动、碰撞时序与相机视角；Proxy Video 与 context views 通过 VAE 编码后作为视频扩散条件。
4. **Plücker 注入：** 将 proxy/context 的相机参数转换为 Plücker rays，经 Camera Encoder 得到 embeddings，在首个 Transformer block 前加到对应 latents；没有相机信息的参考物体图使用 learned embedding。官方未披露 ray encoder 源码或张量分辨率。
5. **训练与延时：** 约 40k 视频/代理视频配对，MiniMax-H3 Ref2VA，124 帧、640×480；流式块 124 帧，重叠固定前缀 34 帧。重建通常 10–20 分钟；8×A800、20 步视频约 1.5 FPS。
6. **局限：** 小物体可能漂浮或幻觉，长期物理变化缺少记忆；未给出独立可复算的机器人策略性能增益。

## 补充几何文献（不是 Mundo 的论文）

- [Cameras as Rays（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/hash/66175cc3d241c1702521dd836cd08585-Abstract-Conference.html) 用于核实 Plücker 坐标 \(r=(d,p\times d)\) 的标准定义。公式属于通用几何背景；Mundo 博客披露的是其在相机条件注入的位置和输入类型，不能由前者推断 Mundo 编码器实现。

## 对 wiki 的映射

- [Mundo 主实体](../../wiki/entities/mundo-world-model.md)
