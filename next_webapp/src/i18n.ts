import { notFound } from "next/navigation";
import { getRequestConfig, type NextRequestConfig } from "next-intl/server";

// Can be imported from a shared config
export const locales = ["en", "cn","es","el"];

export default getRequestConfig(async ({locale}: NextRequestConfig) => {
  // Validate that the incoming `locale` parameter is valid
  if (!locales.includes(locale as any)) notFound();

  return {
    locale,
    messages: (await import(`../messages/${locale}.json`)).default,
  };
});
