import js from "@eslint/js";
import globals from "globals";

export default [
  { ignores: ["docs/exports/**", "exports/**"] },
  {
    files: ["docs/main.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: globals.browser,
    },
    rules: {
      ...js.configs.recommended.rules,
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
    },
  },
];
