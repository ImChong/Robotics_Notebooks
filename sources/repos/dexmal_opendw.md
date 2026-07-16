# dexmal/opendw（OpenDW · DW05）

> 来源归档

- **标题：** OpenDW — DW05 多模态具身世界模型（官方开源）
- **类型：** repo
- **组织：** Dexmal（大晓智能）
- **代码：** <https://github.com/dexmal/opendw>
- **权重（Base）：** <https://huggingface.co/Dexmal/DW05-Base>（Apache-2.0；32D action/proprio；训练 step 140000）
- **权重（RobotWin SFT）：** <https://huggingface.co/Dexmal/DW05-Robotwin>（RoboTwin 2.0 微调；含 `norm_stats.json` 与在线 demo 配置）
- **运行时依赖：** Dexbotic DW05（`pip install -e .`；README 亦提及 `gitlab.dexmal.com/robotics/dexbotic-open` 历史路径）
- **入库日期：** 2026-07-16
- **一句话说明：** **DW05** 是 Dexmal 开源的 **动作条件具身世界模型**：在 **Wan 骨干 + MoT 三专家头（video / action / value）** 上联合 **未来视频预测、动作生成与状态–价值估计**；提供 **DW05-Base** 通用 32 维 checkpoint 与 **DW05-Robotwin** 下游评测包，数据管线对齐 **RoboTwin 风格 JSONL**。

## 架构要点（README 摘要）

| 模块 | 说明 |
|------|------|
| **输入** | 语言、图像/视频、机器人类型、状态、动作 |
| **骨干** | **Wan** 视频扩散系 backbone |
| **MoT 专家** | **Video Expert**（未来视频）、**Action Expert**（动作 chunk）、**Value Expert**（状态–价值；后续版本更新） |
| **监督类型** | 样本 `type` 可为 `action`、`wm` 或二者兼有——同一 JSONL 帧可混训策略与世界建模 |
| **动作接口** | **action_dim=32**、**proprio_dim=32**；checkpoint 默认 **bfloat16** |

## 权重与运行时 bundle（HF 卡摘要）

**DW05-Base** 为通用基础 checkpoint，**不含** RobotWin policy 归一化统计；下游策略推理须自配或与 checkpoint 匹配的 `norm_stats.json`。

```
DW05-Base/
  model.pt
  vae/model.pth
  text_encoder/model.pth
  tokenizer/...
```

加载示例（Dexbotic）：

```python
from dexbotic.model.dw05 import DW05ModelConfig

model_cfg = DW05ModelConfig(
    load_text_encoder=True,
    skip_dit_load_from_pretrain=True,
    action_dim=32,
    proprio_dim=32,
)
model = model_cfg.build_model(model_dtype=torch.bfloat16, device="cuda:0")
model.load_checkpoint("/path/to/DW05-Base/model.pt")
```

环境变量：`DW05_MODEL_BASE_PATH` 指向 bundle 根目录（无需复现上游 cache 目录名）。

## 数据格式（RobotWin-style JSONL）

每 episode 一个 JSONL；每行一帧，典型字段：

- 三视角 `images_1/2/3`（video 或 image URL + `frame_idx`）
- `type`: `["action","wm"]` 等
- `robot.prompt` / `robot.state` / 可选 `proprio` / action 轨迹
- `worldmodel.caption`（视频样本）
- 缓存 `text_embeddings/` 与可选 `norm_stats.json`

内置 recipe：`robotwin_baseline`（`dexbotic/data/dataset/dw05/data_source.py`）。

## 推理入口（README）

| 场景 | 入口 |
|------|------|
| 动作条件 RobotWin 在线 demo | `playground/online_demos/robotwin_online_demo.py` |
| 单图开环视频生成 | `playground/example_dw_exp.py --task inference` |
| JSONL 帧 + action/proprio 条件 | `script/dw/infer_aloha_joint.py` |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Dexmal DW05](../../wiki/entities/dexmal-dw05.md) | 实体归纳页：MoT 联合 WAM、权重分工、数据与推理 |
| [Dexmal DM0.5](../../wiki/entities/dexmal-dm05.md) | 同机构 **VLA 基础模型** 线；DW05 偏 **世界模型 + 动作条件视频** |
| [World Action Models](../../wiki/concepts/world-action-models.md) | **Joint WAM** 族：共享 Wan 骨干同时服务视频/动作/价值头 |
| [τ₀-World Model](../../wiki/entities/tau0-world-model.md) | 同属 **Wan 系联合视频–动作**；τ₀-WM 强调异构掩码与测试时 propose–evaluate–revise |
| [RoboTwin 2.0](../../wiki/entities/robotwin.md) | DW05 数据注解、SFT checkpoint 与在线 demo 对齐 RobotWin 管线 |
| [mimic-video（VAM）](../../wiki/methods/mimic-video.md) | 另一条视频–动作路线；DW05 为 **端到端开源权重 + 训练代码** |

## 第三方组件与许可

- 发布许可：**Apache-2.0**
- 上游组件含 **Wan2.2**、**uMT5** 兼容 tokenizer/text encoder（各自上游许可须保留 NOTICE）

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/dexmal-dw05.md`](../../wiki/entities/dexmal-dw05.md)**；HF Base 卡要点已并入本 source，不单独建 sites 页。
