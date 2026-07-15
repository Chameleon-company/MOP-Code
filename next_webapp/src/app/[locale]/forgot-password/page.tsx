"use client";

import React, { useState } from "react";
import { Link } from "@/i18n-navigation";

const ERROR_MESSAGES: Record<string, string> = {
  MISSING_FIELDS: "Please enter your email address.",
  INVALID_EMAIL: "Please enter a valid email address.",
};

const ForgotPasswordPage = () => {
  const [email, setEmail] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [success, setSuccess] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!email) {
      setError("Please enter your email address.");
      return;
    }
    setError("");
    setLoading(true);

    try {
      const response = await fetch("/api/auth/forgot-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();
      if (response.ok) {
        setSuccess(true);
      } else {
        setError(
          ERROR_MESSAGES[data.code] ||
            data.message ||
            "Something went wrong. Please try again.",
        );
      }
    } catch {
      setError("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center relative"
      style={{ backgroundImage: "url('/img/mainImage.png')", backgroundSize: 'cover', backgroundPosition: 'center' }}
    >
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />

      <div className="relative z-10 w-full max-w-lg mx-4">
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-10 sm:p-12">
          {/* Logo */}
          <div className="flex justify-center mb-6">
            <img
              src="/img/new-logo-green.png"
              alt="Melbourne Open Data logo"
              className="h-16 w-auto"
            />
          </div>

          {/* Heading */}
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white text-center mb-1">
            Forgot Password
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 text-center mb-8">
            Enter your email and we&apos;ll send you a temporary password
          </p>

          {error && (
            <div className="rounded-lg bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 p-3 text-sm mb-5">
              {error}
            </div>
          )}

          {success && (
            <div className="rounded-lg bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-800 text-green-600 dark:text-green-400 p-3 text-sm mb-5">
              If this email exists, a temporary password has been sent. Please check your inbox and follow the link in the email to reset your password.
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate className="space-y-5">
            <div>
              <label
                htmlFor="email"
                className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1"
              >
                Email
              </label>
              <input
                type="email"
                id="email"
                name="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-xl border border-gray-300 px-4 py-3 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-500"
              />
            </div>

            <button
              type="submit"
              disabled={loading || success}
              className="w-full bg-green-600 hover:bg-green-700 disabled:opacity-60 text-white font-semibold py-3.5 rounded-lg transition mt-6"
            >
              {loading ? "Sending..." : "Send Reset Link"}
            </button>
          </form>

          <p className="mt-6 text-sm text-gray-600 dark:text-gray-400 text-center">
            <Link
              href="/login"
              className="text-green-600 hover:text-green-700 font-medium"
            >
              Back to Sign In
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default ForgotPasswordPage;
