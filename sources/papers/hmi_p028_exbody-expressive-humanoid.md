# Expressive Whole-Body Control for Humanoid Robots（ExBody，HMI P028）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Expressive Whole-Body Control for Humanoid Robots
- **短名：** ExBody
- **类型：** paper / hmi-papers / 动作跟踪与全身控制
- **HMI ID：** P028
- **年份：** 2024
- **原文：** https://arxiv.org/abs/2402.16796
- **代码：** https://github.com/chengxuxin/expressive-humanoid
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 重点跟踪上身表达参考，同时用速度命令与鲁棒奖励约束下肢，使真实人形在保留表现力时仍可部署行走。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P028](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P028.md)

## 开源状态（步骤 2.5）

- **结论：** 已开源（expressive-humanoid）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

把整段人体动作逐关节重定向给人形机器人，腿部往往最先出问题：人体腿长、质量分布和接触时序与机器人不同，精确追腿会牺牲平衡。ExBody做了一个很重要的任务拆分：上半身继续追踪表达动作，腿部不追人体腿轨迹，只要求机器人根部完成速度和朝向目标，让RL自行找到适合本体的落脚。

**对 wiki 的映射：** [`wiki/entities/paper-exbody-expressive-humanoid.md`](../../wiki/entities/paper-exbody-expressive-humanoid.md)

### 摘录 2

ExBody先从CMU动作库中整理约780段人体动作，再把人体骨架映射到19自由度Unitree H1。人体肩、髋等球形关节不能直接复制给由多个转动关节组成的机器人，作者使用指数映射把球形旋转分解到机器人对应关节。这个重定向结果并不会作为全身逐帧参考直接交给策略，而是被拆成两类命令：上肢表达目标保留手臂和躯干的动作语义，根部移动目标只描述机器人整体应该怎样移动。

**对 wiki 的映射：** [`wiki/entities/paper-exbody-expressive-humanoid.md`](../../wiki/entities/paper-exbody-expressive-humanoid.md)

### 摘录 3

上肢目标包含九个上半身关节目标和18维关键点位置，用来表达挥手、摆臂、舞蹈等身体动作。根部目标包含三维线速度、身体滚转与俯仰、偏航方向以及身体高度。这样，同一段上肢动作可以重新组合不同的站立、前进和转向命令；腿部不必复制人体参考中的每一次屈膝和落脚，而是根据机器人自身比例、质量和接触状态寻找可执行步态。

**对 wiki 的映射：** [`wiki/entities/paper-exbody-expressive-humanoid.md`](../../wiki/entities/paper-exbody-expressive-humanoid.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-exbody-expressive-humanoid.md`](../../wiki/entities/paper-exbody-expressive-humanoid.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
