"use client"; 

import React from "react";
import useThemeStore from "../zustand/store";

export default function ThemeProvider({ children }: { children: React.ReactNode }) {
  const theme = useThemeStore((state) => state.theme); // Access Zustand theme

  return (
    <div className={theme}>
      {children}
    </div>
  );
}