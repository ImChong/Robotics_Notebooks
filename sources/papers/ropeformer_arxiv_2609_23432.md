# RopeFormer: Cross-Trial Adaptation from Interaction History for Dynamic Rope Manipulation（arXiv:2609.23432）

> 来源归档（ingest）

- **标题：** RopeFormer: Cross-Trial Adaptation from Interaction History for Dynamic Rope Manipulation
- **类型：** paper / manipulation / deformable-objects / rope / reinforcement-learning / transformer
- **arXiv abs：** <https://arxiv.org/abs/2609.23432>
- **PDF：** <https://arxiv.org/pdf/2609.23432>
- **项目页：** <https://ropeformer.github.io/> — 归档见 [`sources/sites/ropeformer-github-io.md`](../sites/ropeformer-github-io.md)
- **代码 / 数据：** **待发布** — 项目页 Code 按钮 **SOON**（2026-09-30）；论文摘要写 *code and data are available at* 项目页
- **机构：** 加州大学伯克利分校（UC Berkeley）、西安交通大学（Xi'an Jiaotong University）、南方科技大学（SUSTech）、北京大学（Peking University）— Menglin Wu、Kaixiang Yao（共一）；Masayoshi Tomizuka、Yuxin Chen
- **入库日期：** 2026-09-30
- **一句话说明：** **Transformer-XL** 在 **trial 间保留** 机器人–绳 action–response 上下文（权重固定、无在线绳参估计）；Newton 多 trial PPO；仿真 384 绳 × 三任务；**H1-2** 真机未见绳 T1→T3 Swing **−30.9%** TAT、Twirl **−33.9%**、Whip **0.2→2.3/3** 命中。

## 核心摘录

### 三任务

| 任务 | 设定 | 主指标 |
|------|------|--------|
| Rope_Swing | 单臂持端，自由端目标角速度旋转 | 达到稳态角速度时间 / 成功率 |
| Rope_Twirl | 双臂持两端，中点绕 hand–hand 轴旋转 | 组合门控 \(G_t>0.30\) 的 acquisition time |
| Rope_Whip | 单臂甩 tip 扫目标线段 | 轨迹法向最大偏差 \(d_{\max}\) |

### 仿真（retain vs reset context，节选）

- **Swing TXL-1 T2–T5：** acquisition **7.39→4.89 s**；success **48.2→79.8%**
- **Twirl TXL-1 T2–T5：** **5.79→4.95 s**（全 384 绳）
- **Whip：** retained context 降低 mean \(d_{\max}\)（角度组 0.07–0.53 cm）

### 真机 H1-2

- **TXL-1 冻结权重**；ZED 2i 单 marker 三角化；30 Hz 策略 / 250 Hz 低层
- **Swing TAT T1→T3：** 8.57→5.92 s（**−30.9%**）
- **Twirl TAT T1→T3：** 3.23→2.13 s（**−33.9%**）；MNE **−55.6%**
- **Whip：** 十组 rope–height，mean hits **0.2→2.3 / 3**

## 对 wiki 的映射

- 新建：[paper-ropeformer](../../wiki/entities/paper-ropeformer.md)
- 交叉：[paper-flying-knots](../../wiki/entities/paper-flying-knots.md)、[manipulation](../../wiki/tasks/manipulation.md)

## 当前提炼状态

- [x] arXiv + 项目页核查（2026-09-30）
- [ ] 代码/data 发布后补 `sources/repos/`
