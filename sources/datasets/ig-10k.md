# IG-10K（The Imitator Game 配对数据集）

> 来源归档（dataset）

- **标题：** IG-10K
- **类型：** dataset / paired human–robot / manipulation
- **Hugging Face：** <https://huggingface.co/datasets/imitator-game/IG-10K-Dataset>
- **仿真资产：** <https://huggingface.co/datasets/imitator-game/IG-10K-Assets>
- **ModelScope：** <https://modelscope.cn/datasets/Zhouxunzhe/IG-10K-Dataset>（资产：`Zhouxunzhe/IG-10K-Assets`）
- **组织：** imitator-game（HKU / TranscEngram / Fudan / ZJU）
- **论文：** <https://arxiv.org/abs/2608.22301>
- **项目页：** <https://imitator-game.github.io/data.html>
- **格式：** LeRobot-0.5.0
- **门控：** 否（截至 2026-09-02 公开可下）
- **入库日期：** 2026-09-02
- **一句话说明：** 目前最大的环境对齐人–机配对操作集，且唯一在仿真与真机同时覆盖 L0–L3 四级。

## 规模（论文 / HF 口径）

| 字段 | 数值 |
|------|------|
| 配对 episode | **20,000+**（真机 11.7K + 仿真 10K） |
| 任务 | **50+** 基础任务 × 四级 ≈ 200+ 变体 |
| 领域 | 家居 12 / 超市 7 / 餐饮 17 / 物流 4 / 医院 6 / 实验室 7 |
| HF 体积 | 约 **747 GB**（Dataset）；Assets 另下到 `~/.maniskill/data` |
| 标注 | 多视角 RGB-D、3D MANO（真机人类 clip）、分割、三层语言 |

## 布局（官方推荐）

```text
demos/
├── demo_data/          # 人类示范
└── imitator_data/      # 机器人（仿真）示范
```

HF 仓目录快照含 `imitator_human_v1`、`imitator_robot_v1`、`imitator_sim_v1_zed2i`。特征与层级处理见官方仓 `examples/baselines/lerobot_dataset/README.md`。

## 采集边界

- 真机：双臂 Realman + VR 遥操作，30 Hz 关节 / 6-DoF。
- 仿真：双臂 Franka + ManiSkill3，OMPL 规划手写航点；复用同一人类 clip。
- 每任务–层级目前只有 **一种** 替换模式，保证跨级监督可比。
- 若干危险/语义不适配任务省略 L3。

## 关联资料

- 论文：[`sources/papers/imitator_game_arxiv_2608_22301.md`](../papers/imitator_game_arxiv_2608_22301.md)
- 仓库：[`sources/repos/the-imitator-game.md`](../repos/the-imitator-game.md)
- Wiki：[`wiki/entities/paper-imitator-game.md`](../../wiki/entities/paper-imitator-game.md)
