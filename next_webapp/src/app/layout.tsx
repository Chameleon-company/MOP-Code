import type { Metadata } from "next";

export const metadata: Metadata = {
  title: { default: "MOP", template: "%s | MOP" },
  description: "City of Melbourne Open Playground",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // no <html> or <body> here – [locale]/layout.tsx provides them
  return children;
}
