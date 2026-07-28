# Khrylx/RFC（Residual Force Control 官方实现）

> 来源归档

- **标题：** Residual Force Control (RFC)
- **类型：** repo
- **来源：** CMU（Ye Yuan，GitHub id: Khrylx）
- **链接：** <https://github.com/Khrylx/RFC>
- **项目页：** <https://www.ye-yuan.com/rfc> — 归档见 [`sources/sites/rfc-ye-yuan.md`](../sites/rfc-ye-yuan.md)
- **入库日期：** 2026-07-28
- **一句话说明：** RFC（NeurIPS 2020）官方实现：MuJoCo 人形角色 + 根部残差外力的动作模仿训练/可视化，附 CMU mocap 预处理数据与 8 组预训练模型（芭蕾×3、backflip、cartwheel、jump kick、side flip、handspring）。
- **沉淀到 wiki：** [`wiki/entities/paper-rfc-residual-force-control.md`](../../wiki/entities/paper-rfc-residual-force-control.md)

---

## 核心定位

RFC 论文官方仓库（**License：免费非商用**）。代码组织沿用 DeepMimic 谱系的 MuJoCo + PPO 动作模仿栈：

| 组件 | 路径 | 说明 |
|------|------|------|
| 训练入口 | `motion_imitation/motion_im.py --cfg 0506 --num_threads <N>` | 按 config 训练 RFC 模仿策略，模型/日志落 `results/motion_im/<cfg>/` |
| 可视化入口 | `motion_imitation/vis_im.py --cfg 0506 --iter 1000` | 回放预训练策略（GUI 快捷键见 README） |
| 配置 | `motion_imitation/cfg/*.yml` | 0506/0507/0513（芭蕾 1–3）、8801（backflip）、9002（cartwheel）、9005（jump kick）、9008（side flip）、9011（handspring） |
| 数据 | `data/cmu_mocap/` | CMU mocap 预处理 pickle 随仓附带；新 clip 用 `data_process/convert_cmu_mocap.py --amc_id 05_06` 转换 |
| 预训练模型 | `results/motion_im/<cfg>/` | 每个 config 对应一组已训练权重 |

## 运行要点（README）

- **依赖：** Python ≥ 3.6（MacOS / Linux 测试）；`pip install -r requirements.txt` + MuJoCo（mujoco-py 时代栈）；建议 `export OMP_NUM_THREADS=1` 提升多线程采样性能。
- **复现路径最短：** 直接 `vis_im.py` 回放预训练模型，再按需 `motion_im.py` 重训。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-rfc-residual-force-control](../../wiki/entities/paper-rfc-residual-force-control.md) | 本仓库对应的论文实体页 |
| [deepmimic](../../wiki/methods/deepmimic.md) | RFC 的方法对位与代码谱系上游（reference motion 模仿 + PPO） |
| [mimickit](../../wiki/entities/mimickit.md) | 同谱系后续统一代码库（DeepMimic/AMP/ASE 等）；RFC 为其同期分支 |
