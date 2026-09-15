# WIYH（Hugging Face 数据集）

> 来源归档（dataset）

- **标题：** World In Your Hands（WIYH）
- **类型：** dataset / multimodal / human-centric / manipulation / tactile / egocentric
- **Hugging Face：** <https://huggingface.co/datasets/tars-robotics/WIYH>
- **组织：** 它石智航（TARS Robotics）
- **论文：** <https://arxiv.org/abs/2512.24310>
- **项目页：** <https://wiyh.tars-ai.com/>
- **Devkit：** <https://github.com/tars-robotics/World-In-Your-Hands>
- **许可：** CC BY-NC-SA 4.0（与 GitHub 仓一致）
- **入库日期：** 2026-09-15
- **一句话说明：** 野外人类中心操作集：~1045 h · 125.4k clips · 100+ skills · 10 场景；多视角 RGB、相机标定、3D 手/腕轨迹、指尖压阻触觉与原子/感知/VLM 标注。

## 规模（论文 Table 1 / 项目页口径）

| 字段 | 数值 |
|------|------|
| 时长 | **~1045** 小时 |
| Clips | **125.4k** |
| 技能 | **100+** |
| 场景 | **10** 类（宴会、洗衣、物流、酒店、办公、超市、工业、清洁等） |
| HF 总体积 | **~36.5 TB**（截至 2026-09-15 HF 页） |
| 模态 | 多视角 RGB、相机标定、3D 手/腕动作、深度、掩码、触觉、原子指令、CoT / VLM 标注 |

## 数据形态

- 原始采集为 Oracle Suite 多模态流；发布含 **HDF5**（`dataset.hdf5`）与 **WorldCode JSON** 导出规范（见 HF README）。
- Devkit `wiyh.py` 提供 HDF5 结构可视化、轨迹与投影校验；`wiyh2lerobot/` 提供 **WIYH → LeRobot** 转换脚本。

## 关联资料

- 论文：[`sources/papers/wiyh_arxiv_2512_24310.md`](../papers/wiyh_arxiv_2512_24310.md)
- 项目页：[`sources/sites/wiyh-tars-ai.md`](../sites/wiyh-tars-ai.md)
- 仓库：[`sources/repos/world-in-your-hands.md`](../repos/world-in-your-hands.md)
- Wiki：[`wiki/entities/paper-wiyh.md`](../../wiki/entities/paper-wiyh.md)
