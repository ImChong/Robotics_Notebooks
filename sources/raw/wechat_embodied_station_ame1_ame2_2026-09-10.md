# 从「下一脚看哪里」到「怎样穿过整片地形」：AME-1 到 AME-2 改变了什么？

来源：https://mp.weixin.qq.com/s/VU_JcNP2FZrITTUoA8IDuA  
作者：具身智能之心  
抓取：WebFetch（2026-09-10）

---

## 基本信息对照

| 项目 | AME-1 | AME-2 |
| --- | --- | --- |
| 论文 | Attention-Based Map Encoding for Learning Generalized Legged Locomotion | AME-2: Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding |
| 机构 | ETH RSL；Disney Research Zurich | ETH RSL |
| 年份 | 2025，Science Robotics | 2026，arXiv v2，under review |
| 机器人 | ANYmal-D、Fourier GR-1 | ANYmal-D、LimX TRON1 |
| Paper | https://arxiv.org/abs/2506.09588 | https://arxiv.org/abs/2601.08485 |
| Code（非官方） | https://github.com/SII-FUSC/AME_Locomotion | https://github.com/Kitjesen/ame2 |
| Project | — | https://sites.google.com/leggedrobotics.com/ame-2 |

## 正文要点（节选）

1. **AME-1 局限：** 本体条件 query 选局部 foothold 强；混合跑酷地形缺「整片地形语境」。
2. **编码器 v2：** global + proprio 共同 query local features（消融支撑）。
3. **感知栈：** 深度→局部高程+方差→里程计融合全局地图；4D $(x,y,z,u)$ 输入策略。
4. **任务接口：** 速度跟踪 → 目标到达；中间轨迹自由度更大。
5. **训练：** AME-1 两阶段 PPO（理想感知→噪声）；AME-2 Teacher（GT 地图）→ Student（在线映射）蒸馏。
6. **未见混合地形：** AME-1 51.2% vs AME-2 teacher 95.2%；student 82.4%。
7. **设计路线：** 局部注意力 → 显式记忆 + 全局语境 + 不确定性 + train=deploy 一致性。
