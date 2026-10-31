# TopoRetarget 官方源码

- **类型：** repo / 灵巧手动作重定向
- **代码：** <https://github.com/wuji-technology/toporetarget>
- **项目页：** <https://toporetarget2026.github.io/TopoRetarget/>（[项目页归档](../sites/toporetarget-github-io.md)）
- **论文：** <https://arxiv.org/abs/2606.16272>（[论文归档](../papers/toporetarget_arxiv_2606_16272.md)）
- **核查日期：** 2026-10-31
- **许可：** MIT；手模型及外部数据须分别核对各自许可。

## 开放内容与入口

仓库提供 CPU 逐帧重定向、数据导入、批处理、可视化以及转笔适配器；内置 `data/pen_spin` 片段。README 指定 Linux x86_64、Python 3.10+、MuJoCo 3.4+。

```bash
git clone --recurse-submodules https://github.com/wuji-technology/toporetarget
cd toporetarget
conda env create -f environment.yml
conda activate toporetarget
toporetarget run --data data/pen_spin --clip pen_spin_029
```

## 源码核对

- `toporetarget/cli.py` 分发 `run` 至 `pipeline`；`pipeline.py` 解析数据集和配置，调用 `python -m toporetarget.solver`，按设置可启动 viewer。
- `solver/run.py` 加载轨迹、调用逐帧求解，进行手指自碰撞轨迹优化，并写出结果和诊断 NPZ。
- `solver/session.py` 的 `FrameSolver` 维护上一帧状态；`solver/retargeter.py` 构建逐帧 interaction graph，依次运行骨方向预热和 mesh SQP。
- `examples/wuji_mjlab/convert_to_wuji_mjlab_pen_spin.py` 只转换单笔、右手 Wuji Hand 2、50 Hz 转笔数据；下游训练使用独立的 [wuji-mjlab](https://github.com/wuji-technology/wuji-mjlab)。

## 对 wiki 的映射

- [TopoRetarget 方法页](../../wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md)
