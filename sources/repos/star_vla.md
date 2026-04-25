# starVLA GitHub Repo

> 来源归档（ingest）

- **项目名称：** starVLA
- **GitHub 地址：** <https://github.com/starVLA/starVLA>
- **核心功能：** 模块化具身智能训练与部署框架。
- **入库日期：** 2026-04-25

## 仓库结构

- `starVLA/`: 核心代码（模型、加载器、训练逻辑）。
- `deployment/`: 真机部署脚本（支持 Franka, ARX5 等）。
- `examples/`: 针对不同基准测试的配置（LIBERO, CALVIN, SimplerEnv）。
- `scripts/run_scripts/`: 训练与评测入口。

## 关键特性实现

### 1. Lego-like Modular Design
- 允许独立进行 **Smoke Tests**。
- 低耦合设计，方便学术界和工业界根据自己的需求替换 VLM 底座或 Action Head。

### 2. 训练支持
- **DeepSpeed ZeRO-2/3**: 支持超大规模参数模型训练。
- **Accelerate**: 方便多卡分布式启动。
- **Co-training Configs**: 提供一键启动 Open X-Embodiment (OXE) 与多仿真环境联合训练的配置文件。

### 3. 部署优化
- 提供了将 VLA 转化为服务（Service）或直接在机器人控制器端集成的 Python API。
- 强调了低延迟推理的重要性。

## 关联 Wiki 页面
- [VLA (Vision-Language-Action)](../../wiki/methods/vla.md)
- [StarVLA (Method)](../../wiki/methods/star-vla.md)
- [RL Frameworks](../../references/repos/rl-frameworks.md)

## 当前提炼状态
- [x] 仓库目录功能
- [x] 核心设计模式
- [ ] 后续：复现该项目的环境搭建步骤
