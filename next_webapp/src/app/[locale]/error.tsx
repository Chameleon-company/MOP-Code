"use client";

import { useEffect } from "react";
import * as Sentry from "@sentry/nextjs";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);

    Sentry.captureException(error, {
      tags: {
        boundary: "locale-error",
      },
      extra: {
        digest: error.digest,
      },
    });
  }, [error]);

  return (
    <div className="flex min-h-screen flex-col bg-white dark:bg-[#1C1C1C]">
      <Header />

      <main className="flex flex-1 items-center justify-center px-4 py-20">
        <div className="text-center">
          <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/20">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-8 w-8 text-red-600 dark:text-red-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v2m0 4h.01M5.07 19h13.86c1.54 0 2.5-1.67 1.73-3L13.73 4c-.77-1.33-2.69-1.33-3.46 0L3.34 16c-.77 1.33.19 3 1.73 3z"
              />
            </svg>
          </div>

          <h2 className="text-3xl font-bold text-gray-900 dark:text-white sm:text-4xl">
            Something went wrong
          </h2>

          <p className="mx-auto mt-4 max-w-md text-base text-gray-500 dark:text-gray-400">
            An unexpected error occurred. Please try again or return to the home
            page.
          </p>

          <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <button
              onClick={reset}
              className="inline-flex items-center justify-center rounded-xl bg-green-600 px-6 py-3 font-medium text-white transition hover:bg-green-700"
            >
              Try again
            </button>

            <a
              href="/"
              className="inline-flex items-center justify-center rounded-xl border border-gray-300 px-6 py-3 font-medium text-gray-700 transition hover:bg-gray-50 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-800"
            >
              Go back home
            </a>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}