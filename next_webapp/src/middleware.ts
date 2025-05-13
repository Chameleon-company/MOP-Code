import createMiddleware from 'next-intl/middleware';

export default createMiddleware({
  locales: ['en', 'el', 'cn', 'es'], // 👈 Add your supported locales
  defaultLocale: 'en'         // 👈 Set your default locale
});

export const config = {
  matcher: ['/', '/(en|fr|de)/:path*'] // 👈 Adjust to match your locale folders
};
