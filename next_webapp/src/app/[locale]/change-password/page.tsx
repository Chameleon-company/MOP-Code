"use client";

import React, { Suspense, useState } from "react";
import { useRouter, Link } from "@/i18n-navigation";
import { useSearchParams } from "next/navigation";
import { Eye, EyeOff } from "lucide-react";

const ERROR_MESSAGES: Record<string, string> = {
  MISSING_FIELDS: "All fields are required.",
  PASSWORDS_DO_NOT_MATCH: "New passwords do not match.",
  PASSWORD_TOO_SHORT: "New password must be at least 8 characters.",
  SAME_AS_TEMP_PASSWORD: "New password must be different from your temporary password.",
  INVALID_CREDENTIALS: "No account found for that email address.",
  INVALID_TEMP_PASSWORD: "Temporary password is incorrect.",
};

function ChangePasswordForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const emailFromQuery = searchParams.get("email") ?? "";

  const [tempPassword, setTempPassword] = useState<string>("");
  const [newPassword, setNewPassword] = useState<string>("");
  const [confirmNewPassword, setConfirmNewPassword] = useState<string>("");

  const [showTemp, setShowTemp] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const [error, setError] = useState<string>("");
  const [success, setSuccess] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);

  const invalidLink = !emailFromQuery;

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");

    if (!tempPassword || !newPassword || !confirmNewPassword) {
      setError("All fields are required.");
      return;
    }
    if (newPassword.length < 8) {
      setError("New password must be at least 8 characters.");
      return;
    }
    if (newPassword !== confirmNewPassword) {
      setError("New passwords do not match.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("/api/auth/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: emailFromQuery,
          temp_password: tempPassword,
          new_password: newPassword,
          confirm_password: confirmNewPassword,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        setSuccess(true);
        setTimeout(() => {
          router.push("/login");
        }, 2000);
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
      style={{ backgroundImage: "url('/img/mainImage.png')", backgroundSize: "cover", backgroundPosition: "center" }}
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
            Change Password
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 text-center mb-8">
            Enter your temporary password and set a new one
          </p>

          {invalidLink && (
            <div className="rounded-lg bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 p-3 text-sm mb-5">
              Invalid or expired reset link. Please request a new password reset.
            </div>
          )}

          {!invalidLink && error && (
            <div className="rounded-lg bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 p-3 text-sm mb-5">
              {error}
            </div>
          )}

          {success && (
            <div className="rounded-lg bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-800 text-green-600 dark:text-green-400 p-3 text-sm mb-5">
              Password changed successfully! Redirecting to login...
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate className="space-y-5">
            {/* Temporary Password */}
            <div>
              <label
                htmlFor="tempPassword"
                className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1"
              >
                Temporary Password
              </label>
              <div className="relative">
                <input
                  type={showTemp ? "text" : "password"}
                  id="tempPassword"
                  placeholder="Enter temporary password"
                  value={tempPassword}
                  onChange={(e) => setTempPassword(e.target.value)}
                  className="w-full px-4 py-3.5 pr-12 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition"
                />
                <button
                  type="button"
                  onClick={() => setShowTemp((v) => !v)}
                  aria-label={showTemp ? "Hide password" : "Show password"}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                >
                  {showTemp ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
            </div>

            {/* New Password */}
            <div>
              <label
                htmlFor="newPassword"
                className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1"
              >
                New Password
              </label>
              <div className="relative">
                <input
                  type={showNew ? "text" : "password"}
                  id="newPassword"
                  placeholder="Enter new password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full px-4 py-3.5 pr-12 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition"
                />
                <button
                  type="button"
                  onClick={() => setShowNew((v) => !v)}
                  aria-label={showNew ? "Hide password" : "Show password"}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                >
                  {showNew ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
            </div>

            {/* Confirm New Password */}
            <div>
              <label
                htmlFor="confirmNewPassword"
                className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1"
              >
                Confirm New Password
              </label>
              <div className="relative">
                <input
                  type={showConfirm ? "text" : "password"}
                  id="confirmNewPassword"
                  placeholder="Confirm new password"
                  value={confirmNewPassword}
                  onChange={(e) => setConfirmNewPassword(e.target.value)}
                  className="w-full px-4 py-3.5 pr-12 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirm((v) => !v)}
                  aria-label={showConfirm ? "Hide password" : "Show password"}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                >
                  {showConfirm ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || success || invalidLink}
              className="w-full bg-green-600 hover:bg-green-700 disabled:opacity-60 text-white font-semibold py-3.5 rounded-lg transition mt-6"
            >
              {loading ? "Updating..." : "Change Password"}
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
}

export default function ChangePasswordPage() {
  return (
    <Suspense>
      <ChangePasswordForm />
    </Suspense>
  );
}
