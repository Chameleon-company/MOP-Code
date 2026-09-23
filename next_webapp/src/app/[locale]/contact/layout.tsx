import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import { getHreflangAlternates } from "@/lib/seo/hreflang";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: "seo" });
  const { canonical, languages } = getHreflangAlternates("/contact");

  return {
    title: t("contact_title"),
    description: t("contact_description"),
    alternates: { canonical, languages },
  };
}

export default function ContactLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
