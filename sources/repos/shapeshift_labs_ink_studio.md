# Shapeshift Labs Ink Studio — 官方源码归档

- **标题：** @shapeshift-labs/ink-studio
- **类型：** repo
- **链接：** https://github.com/siliconjungle/shapeshift-labs-ink-studio
- **入库日期：** 2026-11-24
- **版本：** package.json 0.1.0；核对时仓库 HEAD 为 `b2e76c70618f2616cdbed048ccc180094dd76969`
- **开放状态：** 已开源，package.json 声明 MIT；包标记 private（防止误发 npm registry），不妨碍从 GitHub 安装。
- **核对范围：** README 安装、建模、数据生命周期、场景 JSON、WebGPU 导入与浏览器相关段落；package.json、examples/sculpt.js、docs/build-profiles.md；index.js 加载入口与 cli/ink-studio.js 的 scene/build/buildPreview/render/save 路径。
- **通读：** 未完成

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

## 待核验

完整 README 和扩展模块文档尚未通读；核心采样/渲染内核尚未深入逐函数核对，未实际运行 npm test 或渲染示例。不报告性能和物理精度。
