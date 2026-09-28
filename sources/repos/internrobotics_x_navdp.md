# InternRobotics/X-NavDP（Hugging Face 资产与 checkpoint）

> 来源归档

- **标题：** InternRobotics/X-NavDP
- **类型：** repo（Hugging Face）
- **链接：** <https://huggingface.co/InternRobotics/X-NavDP>
- **论文：** <https://arxiv.org/abs/2607.28560>
- **代码入口：** 与 <https://github.com/InternRobotics/NavDP> 的 `baselines/x-navdp` 自包含 baseline 配套（非独立 GitHub 训练仓）
- **许可：** MIT（X-NavDP 代码；第三方见 `THIRD_PARTY_NOTICES.md`）
- **入库日期：** 2026-09-28
- **一句话说明：** 发布 `navigation_metadata`、机器人 USD、低层控制器 checkpoint、`scene_split.json`、NavDP 预训练与 X-NavDP 后训练权重；场景数据来自 InternRobotics/Scene-N1。
- **开源状态：** **已开源**（权重 + 目录规范；大场景需另下 Scene-N1 / GRScenes100）

## 资产布局（摘要）

| 内容 | 说明 |
|------|------|
| `pretrain_model/navdp_pretrained.ckpt` | NavDP 初始化 |
| `checkpoints/x-navdp_posttrain.ckpt` | GQRM 后训练权重 |
| `data/robots/` | Dingo / G1 USD 等 |
| 场景 | 指向 `SCENE_DIR`（clutter / internscenes 等） |

## 对 wiki 的映射

- [paper-x-navdp](../../wiki/entities/paper-x-navdp.md)
- [NavDP 仓库](./navdp.md)
- [x-navdp 项目页](../sites/x-navdp-project-page.md)
