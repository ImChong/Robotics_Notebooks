# React Bits：React 动效组件集合

- **类型：** repo
- **URL：** <https://github.com/DavidHDev/react-bits>
- **项目页：** <https://reactbits.dev>；[站点核查](../sites/react-bits.md)
- **维护者：** David Haz（DavidHDev）
- **入库日期：** 2026-10-02
- **一句话说明：** 按组件复制或通过注册表安装的 React 动效素材，包含文字、背景、交互组件与微动效。

## 核查资料

- [README](https://github.com/DavidHDev/react-bits/blob/main/README.md)
- [LICENSE.md](https://github.com/DavidHDev/react-bits/blob/main/LICENSE.md)
- [package.json](https://github.com/DavidHDev/react-bits/blob/main/package.json)
- [BlurText 实现](https://github.com/DavidHDev/react-bits/blob/main/src/content/TextAnimations/BlurText/BlurText.jsx)

## 来源摘录归纳

1. README 描述 200+ 组件，分为文字动效、动画、组件、Micro 和背景；这是入库时 main 分支的描述，不是固定版本保证。
2. 每个组件有 JS-CSS、JS-TW、TS-CSS、TS-TW 四种变体；支持手动复制以及 shadcn / jsrepo 安装。
3. README 示例：`npx shadcn@latest add @react-bits/BlurText-TS-TW`。本次只核查文档与源码，没有实际安装组件。
4. `package.json` 的根项目为 private 文档站/注册表工程，使用 React、Vite、jsrepo；消费组件不等于安装整个文档站的依赖。
5. 已核查的 `BlurText.jsx` 使用 `motion/react` 与 `IntersectionObserver`：进入视口后按词/字符分段错峰改变模糊、透明度和位移，并断开 observer。
6. README 还列出 Background Studio、Shape Magic、Texture Lab 等站点工具；没有因此推断它们都可离线部署。

## 开放与许可核查

**源码公开，但采用带限制的 MIT + Commons Clause，而非标准 MIT。** 当前 LICENSE.md 允许组件作为应用、网站或产品的一部分使用（含商业用途），要求保留版权与许可声明，并限制出售、再许可或再分发组件本身，包括组件包和移植版本。归档以当前文件内容为准；不能把“源码公开”写成无限制开源授权。

本条是前端组件工具，不涉及机器人训练代码、权重或数据集。

## 对 wiki 的映射

- [React Bits](../../wiki/entities/react-bits.md) — 机制、安装、机器人展示用途与适用边界。
- [GSAP Skills](../../wiki/entities/gsap-skills.md) — 动画设计规约与现成组件的分工。
