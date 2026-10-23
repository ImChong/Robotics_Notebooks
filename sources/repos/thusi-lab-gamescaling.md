# THUSI-Lab/GameScaling 仓库归档

- **标题：** Scaling in Games: Continued Pre-Training for Embodied Agents in Diverse Virtual Worlds
- **类型：** repo
- **链接：** https://github.com/THUSI-Lab/GameScaling
- **项目页：** https://gamescaling.github.io/
- **论文：** https://arxiv.org/abs/2610.28698
- **模型：** https://huggingface.co/collections/THUSI-Lab/gamescaling-model
- **许可证：** Apache-2.0（代码与权重；不覆盖游戏内容）
- **入库日期：** 2026-10-23
- **一句话说明：** 提供 GameScaling 检查点的推理客户端、提示词模板、部署脚本、示例和测试。

## 仓库核查（2026-10-23）

README 说明模型在键盘/鼠标游戏轨迹上继续预训练。策略调用读取当前游戏帧及短历史，输出约 200 ms 的键鼠动作。仓库提供推理客户端，不包含训练代码；训练语料也未公开。

- **推理入口：** `gamescaling/policy.py` 提供客户端；`gamescaling/action.py` 负责动作格式与解析；示例位于 `examples/run_example.py`。
- **服务方式：** `scripts/serve_vllm.sh` 启动 vLLM；README 列出检查点部署步骤。
- **检查点：** 0.8B、2B、9B、27B 四档，各有 S1（Minecraft 单一世界）和 S3（多游戏混合）版本。
- **开放边界：** 训练实现与 gameplay training corpus 未公开。
- **对应来源：** [项目页](../sites/gamescaling-project.md)；[论文](../papers/gamescaling_arxiv_2610_28698.md)。
- **Wiki 映射：** [GameScaling](../../wiki/entities/paper-gamescaling.md)。
