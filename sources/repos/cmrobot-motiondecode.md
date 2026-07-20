# CMRobot/MotionDecode

> 来源归档

- **标题：** ChingMu MotionDecode 具身运动数据集（Hugging Face）
- **类型：** dataset / repo
- **链接：** https://huggingface.co/datasets/CMRobot/MotionDecode
- **关联项目页：** https://en.chingmu.com/（[MotionDecode 新闻](https://en.chingmu.com/company-news/10746.shtml)）
- **联系：** MotionDecode@chingmu.com
- **入库日期：** 2026-07-20
- **一句话说明：** 青瞳视觉发布的 **千小时级光学动捕** 具身数据集公开入口：含全身骨架、手指、物体 6D、多视角视频与语义标签；`samples/` 提供 **Unitree G1 重定向 CSV**，完整库可通过 HF Request access 或官网申请获取。

## 规模与模态（据 HF README，2026-07-20 核查）

| 维度 | 公开表述 |
|------|----------|
| 时长 | **1000+ h** @ **120 Hz**（总储备叙事见官网 3000h） |
| 场景 | **15+** 真实场景（工业、家庭、零售、医疗、物流、农业、表演等） |
| 任务 | **500+** 标准化任务 |
| 物体 | **200+** 道具 6D 位姿 |
| 模态 | Skeleton · Finger · Object 6D · Video · Labels |

## 数据格式

| 组件 | 格式 | 说明 |
|------|------|------|
| 原始运动 | `.bvh` | Y-up，120 fps，ZYX 旋转，cm，47–67 关节 |
| 重定向轨迹 | `.csv` | 根位置 (m)、四元数、关节角 (rad) |
| 物体 6D | `.csv` | 位置 + 四元数，120 Hz |
| 多视角视频 | `.mp4` | 4–8 相机，时间对齐 |
| 语义标签 | `.jsonl` | 任务 / 场景 / 动作 / 物体 |

## 机器人向发布

- HF 宣称 **100 h** 已重定向至 **Unitree G1**，CSV 位于 `samples/` 目录；使用需注明来源 Chingmu。
- 预期用途：模仿学习、灵巧操作、运动生成、**Sim-to-real 验证**（经重定向轨迹接 MuJoCo 等）。

## 获取方式

- **样例 / 索引：** HF 仓库可直接浏览 `metadata` config 与 `samples/`；`huggingface_hub` 下载示例见 README。
- **完整数据：** HF 页 **Request access**、Discord 社区（国际用户）、官网问卷或邮件申请（与 [`chingmu.md`](../sites/chingmu.md) 交叉引用）。

## 对 wiki 的映射

- [青瞳视觉 CHINGMU](../../wiki/entities/chingmu.md)
- [Unitree G1](../../wiki/entities/unitree-g1.md) — 预重定向目标机型
- [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)
