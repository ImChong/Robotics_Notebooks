# Mondo-Robotics/open-beni

> 来源归档

- **标题：** Open-BENI
- **类型：** repo
- **来源：** Mondo Robotics
- **链接：** https://github.com/Mondo-Robotics/open-beni
- **项目页：** https://huggingface.co/spaces/mondo-robotics/open-beni
- **许可证：** MIT（README 声明；依赖项保留各自许可证）
- **入库日期：** 2026-10-16
- **一句话说明：** BENI 轮式双足机器人的 mjlab 强化学习训练、恢复与跳跃任务，以及 MuJoCo + ONNX 浏览器游戏。
- **沉淀到 wiki：** 是 → [Open-BENI](../../wiki/entities/open-beni.md)

---

## 仓库内容

- python/beni_rl/：任务注册、机器人与执行器配置、速度跟踪/跳跃/恢复环境、PPO 配置、训练、回放及模型导出代码。
- assets/beni_base/：BENI 的 URDF、MJCF、网格与纹理。
- assets/motions/、assets/models/：参考运动数据与仓库提供的策略文件。
- web/：MuJoCo + ONNX 浏览器游戏、客户端和多人游戏服务器。
- beni_rl.headless_export：将 PyTorch checkpoint 导出为 ONNX 与 MNN。

## 官方运行入口

环境建议为 Linux x86_64、Python 3.12、uv；GPU 训练要求支持 CUDA 12.8 的 NVIDIA 驱动，README 建议至少 8 GB 显存。依赖可用 uv sync --locked 安装。

```bash
uv run python -m beni_rl.train Beni-Jump-Flat \
  --env.scene.num-envs 4096 --agent.max-iterations 15000 --agent.seed 42
```

README 也提供 4 个环境、10 次迭代的轻量训练检查，以及通过 Viser 回放、ONNX/MNN 导出和浏览器游戏启动步骤。训练与回放配置以仓库当前 README 为准。

## 开源状态与边界

官方 README 声明训练源码、机器人资产、策略文件和网页游戏均在仓库中，仓库采用 MIT 许可证。Space 是该仓库的在线部署版本。依赖（包括 mjlab 与 MuJoCo Warp）仍分别遵循各自许可证；仓库提供的训练和仿真入口本身不构成真机安全或硬件验证结论。

## 关联归档

- [Open-BENI 项目页与在线演示](../sites/open-beni.md)
- [Open-BENI wiki 实体](../../wiki/entities/open-beni.md)
- [官方 GitHub 仓库](https://github.com/Mondo-Robotics/open-beni)
- [官方 README](https://github.com/Mondo-Robotics/open-beni/blob/main/README.md)
