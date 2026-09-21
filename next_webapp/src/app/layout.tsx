import type { Metadata } from "next";
import "./[locale]/globals.css";

const APP_URL = process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000";

const DEFAULT_DESCRIPTION =
  "Melbourne Open Playground turns Melbourne's open data into interactive tools, use cases, and insights for a smarter, more sustainable city.";

export const metadata: Metadata = {
  metadataBase: new URL(APP_URL),
  title: {
    default: "Melbourne Open Playground",
    template: "%s | Melbourne Open Playground",
  },
  description: DEFAULT_DESCRIPTION,
  openGraph: {
    siteName: "Melbourne Open Playground",
    type: "website",
    title: "Melbourne Open Playground",
    description: DEFAULT_DESCRIPTION,
    images: [
      {
        // NOTE: placeholder default OG image — reuses the existing homepage hero
        // asset (public/img/mainImage.png, 1366x768). Not a purpose-built 1200x630
        // OG asset; consider swapping for a dedicated social-share image later.
        url: "/img/mainImage.png",
        width: 1366,
        height: 768,
        alt: "Melbourne Open Playground",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Melbourne Open Playground",
    description: DEFAULT_DESCRIPTION,
    images: ["/img/mainImage.png"],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
