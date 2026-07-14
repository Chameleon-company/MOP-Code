import Header from "@/components/Header";
import Footer from "@/components/Footer";
import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col bg-white dark:bg-[#1C1C1C]">
      <Header />

      <main className="flex flex-1 items-center justify-center px-4 py-20">
        <div className="text-center">
          <p className="text-8xl font-extrabold text-green-600 dark:text-green-400">404</p>
          <h1 className="mt-4 text-3xl font-bold text-gray-900 dark:text-white sm:text-4xl">
            Page not found
          </h1>
          <p className="mt-4 text-base text-gray-500 dark:text-gray-400 max-w-md mx-auto">
            Sorry, we couldn&apos;t find the page you were looking for. It may have been moved or deleted.
          </p>
          <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3">
            <Link
              href="/"
              className="inline-flex items-center justify-center rounded-xl bg-green-600 px-6 py-3 text-sm font-semibold text-white shadow-sm hover:bg-green-700 transition-colors"
            >
              Go back home
            </Link>
            <Link
              href="/contact"
              className="inline-flex items-center justify-center rounded-xl border border-gray-300 bg-white px-6 py-3 text-sm font-semibold text-gray-700 hover:bg-gray-50 transition-colors dark:border-gray-700 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
            >
              Contact support
            </Link>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}