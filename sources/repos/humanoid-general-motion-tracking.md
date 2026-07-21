# humanoid-general-motion-tracking（GMT 官方代码）

> 来源归档

- **标题：** humanoid-general-motion-tracking
- **类型：** repo
- **来源：** Zixuan Chen / UCSD × SFU（GMT 论文配套）
- **链接：** <https://github.com/zixuan417/humanoid-general-motion-tracking>
- **项目页：** <https://gmt-humanoid.github.io/>
- **论文：** arXiv:[2506.14770](https://arxiv.org/abs/2506.14770) — 归档见 [`sources/papers/gmt_arxiv_2506_14770.md`](../papers/gmt_arxiv_2506_14770.md)
- **入库日期：** 2026-07-21
- **一句话说明：** GMT 官方仓：Unitree G1 上 **MuJoCo sim2sim** 运动跟踪评测；含 **预训练 checkpoint** 与若干示例 motion；**训练与数据处理代码截至入库日未完整开源**。
- **沉淀到 wiki：** [`wiki/entities/paper-gmt.md`](../../wiki/entities/paper-gmt.md)

---

## 核心定位

官方 README 定位为 **轻量、易用的 MuJoCo 仿真测试环境**（Linux / M1 macOS 已测），用于验证预训练策略在参考动作上的跟踪表现，而非完整 IsaacGym 训练栈复现。

---

## 公开内容（README / 目录，截至入库日）

| 组件 | 说明 |
|------|------|
| `sim2sim.py` | 加载 `pretrained.pt` + motion，跑策略驱动的 MuJoCo 物理仿真 |
| `view_motion.py` | 仅运动学回放参考 motion（不跑策略） |
| `assets/motions/` | 示例 `.pkl`（如 `walk_stand.pkl` 等） |
| `assets/robots/g1/` | G1 MJCF/URDF 等多 DoF 配置 |
| Checkpoint | 预训练策略权重（README 称 `pretrained` / 仓内对应文件） |
| News（未勾选） | Data processing and retargeter code will be released soon |

### 最小运行入口（README）

```bash
conda create -n gmt python=3.8 && conda activate gmt
pip3 install torch torchvision torchaudio
pip install "numpy==1.23.0" pydelatin tqdm opencv-python ipdb imageio[ffmpeg] mujoco mujoco-python-viewer scipy matplotlib
python sim2sim.py --robot g1 --motion walk_stand.pkl
python view_motion.py --motion walk_stand.pkl
```

### 开源边界

- **已开源：** sim2sim 推理、可视化、预训练权重、示例动作与机器人资产。
- **未开源 / 待发布：** Adaptive Sampling + Motion MoE 的 **IsaacGym 训练**、完整 **AMASS/LAFAN1 策展与重定向** 管线。
- **免责：** README Alert 标明真机部署因机差异可能失败，仅供研究。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [ResMimic](./resmimic.md) | 下游：**GMT 先验 + 物体条件残差** loco-manipulation |
| [egm](../papers/egm_arxiv_2512_19043.md) | 同族「通用全身 tracking」；Bin 采样 + CDMoE 对照 GMT Adaptive Sampling / MoE |
| [ExBody2 / ASAP 等](../papers/loco_manip_161_survey_009_gmt.md) | 论文 Table 1 对照对象；本仓提供 GMT 侧可跑 sim2sim 锚点 |

## 对 wiki 的映射

- 实体页：[GMT（论文）](../../wiki/entities/paper-gmt.md)
- 流水线：[Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)
- 选型：[人形运动跟踪方法选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)
