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
  const { canonical, languages } = getHreflangAlternates("/ev-infrastructure");

  return {
    title: t("ev_infrastructure_title"),
    description: t("ev_infrastructure_description"),
    alternates: { canonical, languages },
  };
}

export default function EvInfrastructureLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
