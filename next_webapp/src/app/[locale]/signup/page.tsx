"use client";

import React, { useState } from "react";
import { useLocale } from "next-intl";
import { useRouter } from "next/navigation";
import { Link } from "@/i18n-navigation";
import { Eye, EyeOff } from "lucide-react";

const SignUpPage = () => {
  const router = useRouter();
  const locale = useLocale();

  const [form, setForm] = useState({ firstName: "", lastName: "", email: "", password: "" });
  const [errors, setErrors] = useState({ firstName: "", lastName: "", email: "", password: "" });
  const [passwordStrength, setPasswordStrength] = useState<string>("");
  const [confirmPassword, setConfirmPassword] = useState<string>("");
  const [confirmPasswordError, setConfirmPasswordError] = useState<string>("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setForm({ ...form, [name]: value });
    if (name === "password") setPasswordStrength(checkPasswordStrength(value));
  };

  const checkPasswordStrength = (password: string): string => {
    const hasUpper = /[A-Z]/.test(password);
    const hasLower = /[a-z]/.test(password);
    const hasNumber = /[0-9]/.test(password);
    const hasSpecial = /[!@#$%^&*]/.test(password);
    const lengthValid = password.length >= 8;
    if (!lengthValid) return "Weak";
    const passed = [hasUpper, hasLower, hasNumber, hasSpecial].filter(Boolean).length;
    if (passed >= 3) return "Strong";
    if (passed === 2) return "Moderate";
    return "Weak";
  };

  const getPasswordStrengthColor = () => {
    switch (passwordStrength) {
      case "Weak":     return "#e74c3c";
      case "Moderate": return "#f39c12";
      case "Strong":   return "#16a34a";
      default:         return "transparent";
    }
  };

  const validateForm = () => {
    const newErrors = {
      firstName: form.firstName ? "" : "Please enter your first name.",
      lastName:  form.lastName  ? "" : "Please enter your last name.",
      email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email) ? "" : "Please enter a valid email address.",
      password: "",
    };
    const { password } = form;
    const hasUpper   = /[A-Z]/.test(password);
    const hasLower   = /[a-z]/.test(password);
    const hasNumber  = /[0-9]/.test(password);
    const hasSpecial = /[!@#$%^&*]/.test(password);
    const lengthValid = password.length >= 8;
    if (!lengthValid) {
      newErrors.password = "Your password must be at least 8 characters long.";
    } else if (!hasUpper || !hasLower || !hasNumber || !hasSpecial) {
      newErrors.password = "Password must include uppercase, lowercase, number, and special character.";
    }
    setErrors(newErrors);
    return Object.values(newErrors).every((err) => err === "");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSuccessMessage("");
    setErrorMessage("");

    if (confirmPassword !== form.password) {
      setConfirmPasswordError("Passwords do not match.");
      return;
    }
    setConfirmPasswordError("");
    if (!validateForm()) return;

    try {
      setIsSubmitting(true);
      const response = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await response.json();
      if (response.ok) {
        setSuccessMessage("Account created! Redirecting to login...");
        setTimeout(() => router.push(`/${locale}/login`), 1500);
      } else {
        setErrorMessage(data.message || "Sign-up failed. Please try again.");
      }
    } catch {
      setErrorMessage("Something went wrong. Please try again later.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center relative py-12"
      style={{ backgroundImage: "url('/img/mainImage.png')", backgroundSize: "cover", backgroundPosition: "center" }}
    >
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />

      <div className="relative z-10 w-full max-w-lg mx-4">
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-10 sm:p-12">
          <div className="flex justify-center mb-6">
            <img src="/img/new-logo-green.png" alt="Melbourne Open Data logo" className="h-16 w-auto" />
          </div>

          <h1 className="text-2xl font-bold text-gray-900 dark:text-white text-center mb-1">
            Create Account
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 text-center mb-6">
            Join us to get started
          </p>

          {successMessage && (
            <div className="mb-5 rounded-xl border border-green-200 bg-green-50 px-4 py-3 text-sm font-medium text-green-700 dark:border-green-800 dark:bg-green-950/40 dark:text-green-300">
              {successMessage}
            </div>
          )}
          {errorMessage && (
            <div className="mb-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-600 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
              {errorMessage}
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate className="space-y-5">
            <div className="flex gap-3">
              <div className="flex-1">
                <label htmlFor="firstName" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                  First Name
                </label>
                <input
                  id="firstName" name="firstName" type="text" placeholder="Jane"
                  value={form.firstName} onChange={handleChange}
                  className="w-full px-4 py-3.5 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition"
                />
                {errors.firstName && <p className="text-red-500 text-xs mt-1">{errors.firstName}</p>}
              </div>
              <div className="flex-1">
                <label htmlFor="lastName" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                  Last Name
                </label>
                <input
                  id="lastName" name="lastName" type="text" placeholder="Doe"
                  value={form.lastName} onChange={handleChange}
                  className="w-full px-4 py-3.5 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition"
                />
                {errors.lastName && <p className="text-red-500 text-xs mt-1">{errors.lastName}</p>}
              </div>
            </div>

            <div>
              <label htmlFor="email" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                Email
              </label>
              <input
                id="email" name="email" type="email" placeholder="you@example.com"
                value={form.email} onChange={handleChange}
                className="w-full px-4 py-3.5 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition"
              />
              {errors.email && <p className="text-red-500 text-xs mt-1">{errors.email}</p>}
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                Password
              </label>
              <div className="relative">
                <input
                  id="password" name="password" type={showPassword ? "text" : "password"} placeholder="••••••••"
                  value={form.password} onChange={handleChange}
                  className="w-full px-4 py-3.5 pr-12 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition"
                />
                <button type="button" onClick={() => setShowPassword((v) => !v)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors">
                  {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
              {errors.password && <p className="text-red-500 text-xs mt-1">{errors.password}</p>}
              {form.password && (
                <div className="mt-2">
                  <div className="w-full h-1.5 rounded bg-gray-200 dark:bg-gray-600 overflow-hidden">
                    <div
                      className="h-full rounded transition-all duration-300"
                      style={{ width: `${Math.min((form.password.length / 10) * 100, 100)}%`, backgroundColor: getPasswordStrengthColor() }}
                    />
                  </div>
                  <p className="text-right text-xs mt-1 text-gray-500 dark:text-gray-400">
                    Strength: <span style={{ color: getPasswordStrengthColor() }}>{passwordStrength}</span>
                  </p>
                </div>
              )}
            </div>

            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                Confirm Password
              </label>
              <div className="relative">
                <input
                  id="confirmPassword" name="confirmPassword" type={showConfirmPassword ? "text" : "password"} placeholder="••••••••"
                  value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full px-4 py-3.5 pr-12 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition"
                />
                <button type="button" onClick={() => setShowConfirmPassword((v) => !v)}
                  aria-label={showConfirmPassword ? "Hide password" : "Show password"}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors">
                  {showConfirmPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
              {confirmPasswordError && <p className="text-red-500 text-xs mt-1">{confirmPasswordError}</p>}
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-green-600 hover:bg-green-700 text-white font-semibold py-3.5 transition mt-6 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {isSubmitting ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                  Creating account...
                </>
              ) : (
                "Sign Up"
              )}
            </button>
          </form>

          <p className="mt-6 text-sm text-gray-600 dark:text-gray-400 text-center">
            Already have an account?{" "}
            <Link href="/login" className="text-green-600 hover:text-green-700 font-medium">
              Sign In
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default SignUpPage;