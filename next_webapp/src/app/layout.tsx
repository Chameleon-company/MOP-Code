import type { Metadata } from "next";
import "./[locale]/globals.css";

export const metadata: Metadata = {
  title: "Melbourne Open Playground",
  description: "Melbourne Open Data",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
