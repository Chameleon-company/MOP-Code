import { Inter } from "next/font/google";
import "./globals.css";

import { NextIntlClientProvider } from "next-intl";
import { getMessages } from "next-intl/server";

const inter = Inter({ subsets: ["latin"] });

export default async function LocaleLayout({
  children,
  params: { locale },
}: {
  children: React.ReactNode;
  params: { locale: string };
}) {
  const messages = await getMessages({ locale });

  return (
    <html lang={locale} suppressHydrationWarning>
      <body
        suppressHydrationWarning
        className={`${inter.className} min-h-screen flex flex-col`}
      >
        <NextIntlClientProvider messages={messages}>
          <div className="flex-1 flex flex-col">{children}</div>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}