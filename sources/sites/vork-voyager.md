# Vork — Voyager 官方项目与技术解读

> 来源归档（ingest）｜一手项目页 / 创始人技术解读

- **类型：** site / agent-native-renderer / 3D / VFX
- **英文标题：** Vork: A headless 3D and VFX engine built for agents
- **中文标题：** Vork：面向 AI Agent 的无界面 3D 与影视特效引擎
- **发布方：** Voyager（博客作者 Ravi Parikh，Co-founder）
- **项目主页：** https://voyager.so/vork
- **官方技术文章：** https://voyager.so/blog/what-is-vork
- **官方代码：** https://github.com/voyager-so/vork
- **许可证：** https://github.com/voyager-so/vork/blob/main/LICENSE
- **研究预览版本：** 0.1.0（2026-11-06 发布；博客 2026-11-07 更新）
- **代码开放状态：** Source-available，非 OSI 意义开源；允许预览版内部和商业使用，禁止未获许可的修改版分发及对外托管渲染服务
- **核对范围：** 已通读官方项目页全部核心内容及作者《What is vork? The headless 3D and VFX engine for agents, and its limits》完整技术解读（定义、限制、性能、许可、尝试方法与 FAQ），逐项对照主仓 README、Benchmark 原表和 LICENSE；源码入口见 [repo 归档](../repos/vork-voyager.md)。
- **通读：** 已完成
- **入库日期：** 2026-11-08

## 官方主张及有证据的边界

1. **Agent 原生：** 无需 GUI；Agent 写 scene JSON / 自定义 Slang 程序，CLI 渲染并检查图像后迭代。不是“仅通过提示词直接生成电影”，外部资产、跟踪与美术判断仍然重要。
2. **小内核：** Rust + Slang，Vulkan / Metal GPU；保留几何与 ray query，开放相机、材质、光传播、film response、模拟和合成程序接口。
3. **百万鸟：** 官方 skim 片长 8 秒 / 192 帧 / 1080p24 / L4 / 四 samples + OIDN，完整视频渲染中位数 **96.942 s**；此前 **243.809 s setup + 166.030 s 低分辨率 warmup** 不在这 97 秒内，资产准备另计。
4. **对比不等质：** 同四 samples 的 EEVEE **773.7–1082.5 s 为八个热帧外推**（约 8–11×），并非 Blender 完整视频实测；16 samples 外推为 1511–1818 s。运动、材质转换、计时边界不同，不能标注“24×同质加速”。
5. **有正反案例：** EEVEE 对照的默认 final 片子 1.14–2.32×，静图 1.15–1.90×；老版 path tracer 对照 Cycles 有 1.99–5.81× 变慢案例。
6. **功能缺口：** 雾/体积不支持，大片透明/内存敏感，水模拟有限；黑洞为 Schwarzschild 数值近似，不是严格相对论求解器。
7. **复现限制：** Blender 5.2.1 对比 helper 当前在相机准备环节失败；仓库的现有重跑脚本与 headline 的 samples/seed/denoise 配置不完全一致；没有独立验证结果。

## 归档关联

- [Vork 官方源码与基准原始资料](../repos/vork-voyager.md)
- [Vork 结构化实体节点](../../wiki/entities/vork.md)

## 对机器人工作流的含义（推论）

适合 Agent 自动化预演镜头、合成或程序化可视化的探索；**不是** 机器人动力学、碰撞/接触和控制训练的已验证替代方案。不能据特效渲染速度推断物理仿真吞吐。