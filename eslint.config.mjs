import js from "@eslint/js";
import globals from "globals";

const sharedRules = {
  ...js.configs.recommended.rules,
  "no-unused-vars": [
    "warn",
    { argsIgnorePattern: "^_", caughtErrorsIgnorePattern: "^_" },
  ],
};

export default [
  {
    ignores: [
      "docs/exports/**",
      "exports/**",
      "docs/vendor/**",
      "media/**/vendor/**",
      "media/**/out/**",
      "media/**/cache/**",
      "media/**/*.built.html",
    ],
  },
  {
    files: ["docs/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: {
        ...globals.browser,
        ...globals.serviceworker,
        module: "readonly",
        d3: "readonly",
        THREE: "readonly",
        katex: "readonly",
        marked: "readonly",
        DOMPurify: "readonly",
      },
    },
    rules: sharedRules,
  },
  {
    files: ["scripts/*.cjs", "tests/*.cjs", "media/**/*.cjs"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "commonjs",
      globals: {
        ...globals.node,
        ...globals.browser,
      },
    },
    rules: sharedRules,
  },
  {
    files: ["media/**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: globals.browser,
    },
    rules: sharedRules,
  },
  {
    files: ["media/**/*.mjs", "media/zero-to-ppo-video/robot3d.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        ...globals.node,
        ...globals.browser,
      },
    },
    rules: sharedRules,
  },
  {
    files: ["workers/**/*.mjs"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        ...globals.serviceworker,
        caches: "readonly",
        crypto: "readonly",
        fetch: "readonly",
        Headers: "readonly",
        Request: "readonly",
        Response: "readonly",
        TextDecoder: "readonly",
        TextEncoder: "readonly",
        URL: "readonly",
        URLSearchParams: "readonly",
        AbortController: "readonly",
        clearTimeout: "readonly",
        setTimeout: "readonly",
      },
    },
    rules: sharedRules,
  },
];
