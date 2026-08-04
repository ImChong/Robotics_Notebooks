# Light-Loco-Parkour: Versatile Perceptive Whole-Body Locomotion via Multi-Skill Distillation（Light Origins, 2026）

> 来源归档（ingest · 项目页 PDF）

- **标题：** Light-Loco-Parkour: Versatile Perceptive Whole-Body Locomotion via Multi-Skill Distillation
- **缩写：** **LightLP** / 站内亦称 **LightParkour**
- **类型：** paper / humanoid / parkour / perceptive-locomotion / distillation / real2sim2real
- **项目页：** <https://light-loco-parkour.github.io/> — 归档见 [`sources/sites/light-loco-parkour-github-io.md`](../sites/light-loco-parkour-github-io.md)
- **PDF：** <https://light-loco-parkour.github.io/paper.pdf>
- **视频：** <https://youtu.be/96Rfm7OmHjY>
- **arXiv：** 入库时 **暂无编号**
- **作者：** Hongming Chen、Zhuoran Li、Hongxi Wang、Jiangpeng Hu、Ziliang Li、Peize Liu、QingRui Zhao、Xuhao Liu、Liang Pan、Ximin Lyu、Yuntao Ma†、Tingxiang Fan†（† robotics team co-leads）
- **机构：** Light Origins（光原点）
- **发表日期（项目页）：** 2026-08-03
- **入库日期：** 2026-08-04
- **一句话说明：** 在自研 Lightbot 0 上，用 Real2Sim2Real 从稀疏人体动作种子扩出地形配对全身技能，再经多专家 DAgger + 转移组 RL + 深度蒸馏，得到**单一机载深度策略**：仅深度 + 速度指令即可在行走 / 攀爬 / vault 间自主切换，无技能标签与运行时运动图。

## 开源状态（步骤 2.5）

- **核查日：** 2026-08-04，打开项目页与 GitHub 组织 `Light-Loco-Parkour`。
- **已发布：** 项目页、PDF、演示视频。
- **未发布：** 训练/推理代码、权重、数据集；无 arXiv。
- **结论：** **确认未开源**。wiki 实体页「源码运行时序图」写 **不适用**。

## 摘录 1：问题与三条贡献

现有人形全身控制常落在两端：**(a)** 跟踪富表达参考但难地形泛化；**(b)** 在线感知识别地形却很少用手臂/躯干做承重接触。LightLP 用**单一可部署策略**同时覆盖感知 locomotion 与全身跑酷技能。

三条贡献（摘要）：

1. **全身感知控制管线：** 把 RL 速度跟踪 locomotion 扩展为含物体交互跑酷技能的同一策略——开阔地跟踪速度，遇障执行全身穿越，之后恢复行走。
2. **稀疏种子 → 地形条件技能：** 把单条动作扩成跨障碍几何的动力学可行、地形配对参考，而非依赖大体量动作库。
3. **奖励驱动自主技能转移：** 仅从深度与指令决定何时调用何种全身技能；无 one-hot 标签、手写状态机或运行时 motion generator。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-light-loco-parkour.md`](../../wiki/entities/paper-light-loco-parkour.md)；对比 [PHP](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)。

## 摘录 2：方法栈（Fig. 2 / §III–VI）

仿真：**IsaacLab**。控制频率 **50 Hz**；动作为关节位置目标 + PD。

| 阶段 | 内容 |
|------|------|
| **(a) RL teachers** | 特权信息训 1 个感知 locomotion teacher + 每技能 1 个 teacher；数据增广把种子扩成地形配对参考（课程抬障，如 climb 种子 **45 cm → 75 cm** ≈ **0.83H**） |
| **Object-Interaction Mimic** | 重定向后物理修复穿透/浮动接触；RSI 式随机帧重启暴露关键相 |
| **(b) Distill & transition** | 多专家 **DAgger** 合成单一 height-scan 策略（学生无技能标签；critic 可有 group one-hot）；再加 **transition group** 用稀疏奖励学 loco↔技能切换（去转移组成功率可掉到 **0%**） |
| **(c) Depth distillation** | height-scan → 循环 **GRU** 深度学生；DAgger+PPO；辅助重建 height-scan；RealSense D435 噪声/延迟模型；蒸馏 14k + fine-tune 1k iter |

观测（部署）：机载深度 + 本体 + 速度指令 \((v_x,v_y,\omega_z)\)。Teacher 用干净 height scan；actor **不**喂学生推不出的特权量（student-informed teacher）。

## 摘录 3：平台与主结果

**Lightbot 0：** 高 **90 cm**、重 **18.9 kg**、**21 DoF**；踝部平行四杆；QDD 力矩档 **45 N·m**（腰/下肢）/ **15 N·m**（臂）；峰值角速度 **9.42 rad/s**；胸前 **RealSense D435**（下倾 30°）+ 骨盆 IMU；板载 **Jetson Orin Nano**（7–25 W，67 INT8 TOPS）。

**Table V（仿真成功率 %，每格 500 trials；节选 Ours / Teacher / 关键消融）：**

| 设定 | Ours | w/o FT | w/o GRU | Teacher |
|------|------|--------|---------|---------|
| climb-and-step 60 cm (0.66H) | **99.2** | 88.4 | 54.0 | 99.9 |
| climb-and-step 75 cm (0.83H) | **33.4** | 17.0 | 0.0 | 98.6 |
| reverse-vault 50 cm | **96.8** | 72.8 | 25.4 | 99.4 |
| speed-vault 50 cm | **93.4** | 61.8 | 23.2 | 99.6 |
| Stepping stones | **99.9** | 34.6 | 0 | 99.9 |
| Stairs (high) | **83.4** | 80.8 | 12.4 | 83.0 |

对照：PHP 在若干 climb/vault 格失败收敛（✗）；BeamDojo / CReF 覆盖稀疏落足/楼梯子集。未见形状：**pommel-horse** 上 reverse/speed-vault 仍 **93.4% / 95.3%**（Table VI）。

**转移消融（Table VII）：** 无 transition group 时 plane+skills = **0%**；加 transition group 后 **98%**；障碍前反向指令可正确远离（command adherence True）。

**真机：** 室内外零样本；含未见鞍马形障碍、木板桥、踏石、高台、室外路缘楼梯；策略自主切换 locomotion↔技能。

## 摘录 4：局限（§VIII）

1. 种子仍需人工对齐动作与障碍 → 扩展技能集的瓶颈。  
2. 转移目前耦合少量离散技能；障碍重叠/紧挨时行为变差。  
3. 固定胸前深度相机：技能中段遮挡与全身操作覆盖不足。

## BibTeX（项目页）

```bibtex
@misc{chen2026lightlocoparkour,
  title  = {{Light-Loco-Parkour}: Versatile Perceptive Whole-Body Locomotion via Multi-Skill Distillation},
  author = {Hongming Chen and Zhuoran Li and Hongxi Wang and Jiangpeng Hu and Ziliang Li and Peize Liu and QingRui Zhao and Xuhao Liu and Liang Pan and Ximin Lyu and Yuntao Ma and Tingxiang Fan},
  year   = {2026},
  url    = {https://light-loco-parkour.github.io/}
}
```
