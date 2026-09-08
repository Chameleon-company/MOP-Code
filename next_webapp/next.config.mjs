import createNextIntlPlugin from "next-intl/plugin";
import { withSentryConfig } from "@sentry/nextjs";

const withNextIntl = createNextIntlPlugin("./src/i18n.ts");

/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    forceSwcTransforms: true,
  },

  typescript: {
    // !! WARN !!
    // Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    // !! WARN !!
    ignoreBuildErrors: true,
  },

  serverExternalPackages: ["@prisma/instrumentation"],

  outputFileTracingRoot: process.cwd(),

  images: {
    formats: ["image/avif", "image/webp"],
    minimumCacheTTL: 86400,

    remotePatterns: [
      {
        protocol: "https",
        hostname: "images.unsplash.com",
      },
      {
        protocol: "https",
        hostname: "*.supabase.co",
        pathname: "/storage/v1/object/public/**",
      },
    ],
  },
};

const configWithIntl = withNextIntl(nextConfig);

export default withSentryConfig(configWithIntl, {
  org: process.env.SENTRY_ORG,
  project: process.env.SENTRY_PROJECT,
  authToken: process.env.SENTRY_AUTH_TOKEN,

  // Upload a broader set of client source maps so production stack traces
  // can resolve back to the original source code.
  widenClientFileUpload: true,

  // Avoid exposing source maps publicly after they have been uploaded.
  sourcemaps: {
    deleteSourcemapsAfterUpload: true,
  },

  // Keep Sentry build output quieter unless debugging the integration.
  silent: true,
});
