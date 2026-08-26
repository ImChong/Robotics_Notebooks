# JIAjindou/PhyFilter

> 来源归档

- **标题：** PhyFilter 官方案例仓
- **类型：** repo
- **代码：** <https://github.com/JIAjindou/PhyFilter>
- **项目页：** <https://scoardyy.github.io/PhyFilter>
- **论文：** [arXiv:2608.22701](https://arxiv.org/abs/2608.22701) — 归档见 [`sources/papers/phyfilter_arxiv_2608_22701.md`](../papers/phyfilter_arxiv_2608_22701.md)
- **入库日期：** 2026-08-26
- **一句话说明：** 四系统案例（四足 Isaac Gym RL、无人机/空中操作 Simulink、加速度 MATLAB）+ 伴随梯度自动学滤波参数。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [PhyFilter 实体页](../../wiki/entities/paper-phyfilter.md) | 方法与泛化数字 |
| [Locomotion](../../wiki/tasks/locomotion.md) | 四足平地训练→未见地形 |
| [Sim2Real](../../wiki/concepts/sim2real.md) | 物理结构换数据规模 |

## 复现入口（四足 README 摘要）

```bash
cd quadruped_case
python3 legged_gym/legged_gym/scripts/train.py --rl_device cuda:0 --sim_device cuda:0 --headless
python3 legged_gym/legged_gym/scripts/play.py
```

PhyFilter 训练需在 `legged_robot.py` 第 896–922 行取消注释。依赖 Isaac Gym 1.0rc4、PyTorch 1.12.1+cu116。

## 开源状态

**已开源** — 各子目录含 README；无人机/空中操作为 MATLAB/Simulink 部署模型。
