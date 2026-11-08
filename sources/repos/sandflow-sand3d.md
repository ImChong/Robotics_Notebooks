# Sandflow-ai/sand3d — Python 3D modeling for LLMs

- **类型：** repo / 参数化 3D 建模 / 机器人资产
- **官方代码：** https://github.com/Sandflow-ai/sand3d
- **许可证：** Apache-2.0（仓库 LICENSE；不覆盖数据集原素材许可）
- **PyPI：** https://pypi.org/project/sand3d/（核验版本 1.6.3；2026-11-06）
- **README：** https://github.com/Sandflow-ai/sand3d/blob/main/README.md
- **文档：** https://github.com/Sandflow-ai/sand3d/blob/main/docs/api.md
- **写作规范：** https://github.com/Sandflow-ai/sand3d/blob/main/docs/guidelines.md
- **数据：** https://huggingface.co/collections/Sandflow/sand3d
- **核查日期：** 2026-11-08
- **核对范围：** 阅读官方 README、docs/guidelines.md 的核心八条规范，直接核查 programs/washing_machine.py、programs/weight_stack.py、src/sand3d/__init__.py、src/sand3d/assembly.py 中 group/assembly/export、src/sand3d/urdf.py 的 Mechanism/write_urdf、src/sand3d/usd.py 的 write_usda、src/sand3d/joints.py 的 pose、src/sand3d/solids.py 的 extrude 定义及调用链；源码审读范围是关键入口与导出实现，并非宣称逐行审阅整个代码库。

## 安装、API 与受支持的输出

- Python 3.11 或 3.13；依赖 Blender `bpy`（官方 README 注明 Python 3.12 无 bpy wheel）。
- `pip install sand3d`；`import sand3d as s3d`。
- `s3d.box / extrude / hole / fillet / at / group / assembly / joint / pose / export` 构造可编辑的几何、装配、关节。
- `s3d.export(object, "x.glb")`：网格、材质、关节 extras（当前 pose）；`.urdf`：各运动件 link、joint 和 limits + OBJ meshes（rest pose）；`.usda`：OpenUSD 文本、UsdGeom/UsdPhysics joint；`.obj` / `.stl`：当前 pose 的网格。
- `.usd` / `.usdc` / `.usdz` **不能直接导出**，官方代码提示先导出 `.usda` 再经 USD 官方工具转换。
- URDF/USD 导出**不提供完整动力学**：URDF writer 为 joint limits 写 effort=0、velocity=0；USD 源码说明无 mass、drive、collider。不能把导出视为已经通过机器人接触仿真测试。

## 已确认的实际调用路径（用于时序图）

1. `programs/washing_machine.py:washer()` 调用 `s3d.box/pocket/cylinder/group` 构件，再调用 `s3d.joint` 记录 door_hinge、drum_spin、drawer_slide、dial_turn，最后调用 `s3d.assembly` 返回 `Handle`。
2. `src/sand3d/__init__.py` 暴露 API 并对公开 verb 做 `_verb` 包装；`assembly.py:assembly` 通过 `group` 返回装配。
3. 使用者调用 `s3d.export(washer(), "washer.urdf")`；`assembly.py:export` 按后缀派发 `urdf.write_urdf`；`write_urdf` 使用 `Mechanism`/rest hierarchy、写 link/joint 和配套 OBJ。
4. 改用 `washer.usda` 时同一 `export` 分派 `usd.write_usda`；该函数复用 `urdf.Mechanism` 生成 USD 结构。
5. `programs/weight_stack.py` 的 `if __name__ == "__main__"` 真实 CLI 入口：`s3d.empty_scene()` → `weight_stack(...)` → `s3d.export(...)`，可用于命令行复现。

## 映射

- [Sand3D 实体页](../../wiki/entities/sand3d.md)
- [官方发布说明](../blogs/sandflow_sand3d_release_2026_11_05.md)
- [官方网站](../sites/sandflow-sand3d.md)
