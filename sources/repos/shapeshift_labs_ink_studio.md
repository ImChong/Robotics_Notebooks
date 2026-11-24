# Shapeshift Labs Ink Studio — 官方源码归档

- **标题：** @shapeshift-labs/ink-studio
- **类型：** repo
- **链接：** https://github.com/siliconjungle/shapeshift-labs-ink-studio
- **入库日期：** 2026-11-24
- **版本：** package.json 0.1.0；核对时仓库 HEAD 为 `b2e76c70618f2616cdbed048ccc180094dd76969`
- **开放状态：** 已开源，package.json 声明 MIT；包标记 private（防止误发 npm registry），不妨碍从 GitHub 安装。
- **核对范围：** 固定版本 README.md 全部 1322 行（包括 License 后的扩展说明）、package.json、examples/sculpt.js、docs/build-profiles.md、docs/adaptive-sculpt-sampling.md、docs/browser-renderer.md 全文；源码核对 index.js 加载、CLI scene/build/buildPreview/render/save、src/kit/k3-shapes.js 的 join、src/sculpt.js 的 bake/bakeBand、src/sculpt-sampling.js 全文、src/builder.js 的数据布局与 finish、cli/diarama-pose.js 的时钟与骨骼路径、cli/render.js 的排序与像素覆盖计算、webgl.mjs 导出。范围为项目核心建模、采样、渲染与运行入口，不宣称穷尽扩展音频/故事等模块。
- **通读：** 已完成

## 主要机制

- 以带符号距离场描述几何，布尔组合与减材雕刻后进行表面采样和光照烘焙，产生用于像素风渲染的高斯 splat。
- 原生 world.data 是 Float32Array，每个 splat 16 个浮点值，包含位置、密度、对称协方差的六个独立元素、打包颜色、类型和动画相关数据。
- 固定 seed、固定前向时间步可用于可重复场景；有状态场景不能任意跳回历史时刻，需重建后推进。
- Node CPU 离屏渲染、默认 WebGL 2 浏览器渲染；另有 WebGPU 纹理/几何图、PLY/SPZ 导入示例和 Gaussian glTF 混合资产导入。可导出 GLB 网格，但不能据此保证物理仿真资产合规。

## 可运行入口

- [README](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/README.md)
- [package.json](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/package.json)
- [index.js](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/index.js)
- [CLI](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/cli/ink-studio.js)
- [雕刻示例](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/examples/sculpt.js)
- [构建模式](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/docs/build-profiles.md)

```bash
npm install github:siliconjungle/shapeshift-labs-ink-studio
npx ink-studio help
npx ink-studio render watcher --t 2.6 --colored
```

## 对 wiki 的映射

- [Ink Studio 实体](../../wiki/entities/shapeshift-labs-ink-studio.md)：可编程几何、可视化与离屏渲染。

## 核心实现与可复核细节

- [sculpt.js](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/src/sculpt.js)：`S.bake()` 构建空间网格，遍历正负固体并进入 `bakeBand`；按复合 SDF 的距离/数值梯度迭代投影候选点，若残差超过采样间距一半则丢弃，按表面所有者选材质并调用 `shadeOwned`。移动整体使用自身 SDF，避免接触处烘焙缺洞。
- [sculpt-sampling.js](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/src/sculpt-sampling.js) 和[自适应采样说明](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/docs/adaptive-sculpt-sampling.md)：默认均匀；启用时以切平面偏离量估计局部弯曲，对不透明表面分别用 1.8 倍和 0.7 倍基础间距，默认曲率阈值 0.15、silhouetteBand 0.22，支持最多 16 个观察方向。透明玻璃等不走该自适应分支；双候选扫描有额外烘焙代价，不能保证所有场景减点。
- [builder.js](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/src/builder.js)：底层按 16 个 Float32 写入，`finish` 以 `data.subarray(0, n * 16)` 输出；原生数据主体每 splat 64 字节，不含材质、动画、索引及 GPU 资源。
- [render.js](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/cli/render.js)：CPU 路径投影协方差得到屏幕椭圆，深度排序后按覆盖/高斯权重混合；之后使用像素外观与调色板控制。不同透明度/笔触路径并非同一个简单高斯公式。
- [diarama-pose.js](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/cli/diarama-pose.js)：静态 world 无 anim 时直接返回；有状态场景以最大 1/60 秒步长向前推进，拒绝倒退，骨骼路径在更新后生成 posed buffer。
- [browser renderer](https://github.com/siliconjungle/shapeshift-labs-ink-studio/blob/b2e76c70618f2616cdbed048ccc180094dd76969/docs/browser-renderer.md)：WebGL 2 与 CPU 共享相机/时钟契约，但像素精度和扩展特性不同；`changedRows` 指 512-splat 纹理行，空数组表示未改变，首次/数量变动全量上传。WebGL 调色板 resolve 限 64 色，超出时普通模式使用前 64 色，strictContract 拒绝不兼容输入。
- README 的物理工具包括 MPM、XPBD、体素材料与事件持久化；它们服务场景创作，不能据此宣称机器人接触动力学或策略训练精度。网格导出与像素渲染是不同路径；常规 mesh-to-splat 导入不导入纹理图像，而独立资产读取/GLB 交换支持纹理。

## 验证边界

官方提供 npm test、selftest、stats、perf、snap/check、passes、environment-parity 和对照样例。已核对命令及源码机制，未实际执行上游测试或性能基准；不能报告实测帧率、机器人仿真精度或像素一致性。回归图像应在相同后端、相机、seed、分辨率及时间下比较。
