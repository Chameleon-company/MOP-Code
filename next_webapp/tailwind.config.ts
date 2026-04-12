// tailwind.config.ts
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        primary: "var(--color-primary)",
        dark:    "var(--color-dark)",
        light:   "var(--color-light)",
        white:   "var(--color-white)",
      },
      fontFamily: {
        sans: ["Poppins", "sans-serif"],
      },
      gridAutoRows: {
        fr: "1fr", // use class `auto-rows-fr` for uniform card heights
      },
    },
  },
  plugins: [
    // If you choose to truncate descriptions:
    // require('@tailwindcss/line-clamp'),
  ],
};

export default config;
