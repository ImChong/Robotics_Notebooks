# HIW-500: Humanoids In-the-Wild Dataset

- **标题:** HIW-500: Humanoids In-the-Wild Dataset
- **类型:** dataset / huggingface
- **链接:** <https://huggingface.co/datasets/BitRobot/HIW-500>
- **项目页:** <https://bitrobot-foundation.github.io/humanoids-in-the-wild-500-hours/>
- **收录日期:** 2026-06-25

## 一句话摘要

BitRobot 与 Unitree、Hugging Face 联合发布的 **最大规模开源人形机器人远程操作数据集之一**：在东南亚 **12 个真实家庭** 中采集 **Unitree G1** 全身遥操作示范，**500+ 小时 / 23K+ 集 / ~10 TB**，覆盖 **10+ 家务任务** 与 **148K+ 细粒度子任务语言标注**；提供原始 ROS bag / MCAP 与（计划中的）LeRobot 格式。

## 规模与覆盖

| 维度 | 数值 |
|------|------|
| 时长 | **500+ 小时** |
| Episodes | **23K+** |
| 数据量 | **~10 TB**（HF 当前约 8.94 TB） |
| 家庭场景 | **12** 个真实家庭（东南亚） |
| 家务任务 | **10+** 类（见下表） |
| 子任务标签 | **161** 类 · **148K+** 条标注 |

### 任务示例（项目页 V1）

- Building children table（搭建儿童桌）
- Hang hanger（挂衣架）
- Clean up the room（整理房间）
- Setting the table（摆桌）
- Restocking fridge（冰箱补货）
- Kitchen organization（厨房整理）
- Hang keys on a hook（钥匙挂钩）
- Move pillow to sofa（抱枕归位）
- Sweep floor（扫地）
- Picking trash（捡垃圾）
- Clothes washing（洗衣）

## 采集硬件与模态

- **机器人：** Unitree G1，**夹爪**末端执行器（非灵巧手）
- **相机：**
  - **头部：** RGB 立体，480p，30 FPS
  - **腕部：** RGB + 立体 IR，480p，30 FPS
- **状态与动作：** 29-DoF 关节、末端执行器、IMU、里程计、全身遥操作动作轨迹
- **元数据：** 语言标注、episode 信息、相机内外参

## 数据格式与访问

| 格式 | 状态 | 入口 |
|------|------|------|
| 原始 ROS bag / MCAP | **已发布** | <https://huggingface.co/datasets/BitRobot/HIW-500> |
| LeRobot 格式 | Coming Soon | 项目页路线图 |

- **许可：** Humanoid Data for Training & Evaluation（项目页）；商用或定制采集需联系 BitRobot。
- **路线图：** V1（2026-06）发布 500+ 小时；V2 计划扩展任务与环境。

## 引用

```bibtex
@misc{hiw500_2026,
  title={HIW-500: Humanoids In-the-Wild Dataset for Robot Learning},
  author={BitRobot and Unitree and Hugging Face},
  year={2026},
  howpublished={\url{https://bitrobot-foundation.github.io/humanoids-in-the-wild-500-hours/}}
}
```

## 对 Wiki 的映射

- **wiki/entities/hiw-500-dataset.md**：数据集实体页（真机全身遥操作 · in-the-wild 家务）
- **wiki/tasks/teleoperation.md**：主流遥操作系统对照表补充
- **wiki/queries/humanoid-training-data-pipeline.md**：真机操作数据来源层交叉引用
