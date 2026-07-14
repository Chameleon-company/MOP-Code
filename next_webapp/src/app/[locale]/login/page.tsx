"use client"

import React, { useState } from "react";
import { useTranslations, useLocale } from "next-intl";
import { useRouter } from "next/navigation";
import { Link } from "@/i18n-navigation";
import { Eye, EyeOff } from "lucide-react";

function LoginForm() {
    const t = useTranslations("login");
    const locale = useLocale();
    const router = useRouter();
    const [email, setEmail] = useState<string>("");
    const [password, setPassword] = useState<string>("");
    const [passwordVisible, setPasswordVisible] = useState(false);
    const [error, setError] = useState<string>("");
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;
        if (name === "email") setEmail(value);
        else setPassword(value);
    };

    const togglePasswordVisibility = () => {
        setPasswordVisible(!passwordVisible);
    };

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!email || !password) {
            setError("Please fill in both fields");
            return;
        }

        try {
            setIsSubmitting(true);
            setError("");

            const response = await fetch("/api/auth/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password }),
            });

            const result = await response.json();

            if (!response.ok) {
                setError(result.message || "Login failed");
                return;
            }

            localStorage.setItem("userId", result.data.userId.toString());
            localStorage.setItem("user", JSON.stringify(result.data));
            localStorage.setItem("token", result.data.token);

            if (result.data.roleId === 1) {
                router.push(`/${locale}/admin/dashboard`);
            } else {
                router.push(`/${locale}/profile`);
            }
        } catch (err) {
            console.error("Login error:", err);
            setError("Something went wrong. Please try again.");
        } finally {
            setIsSubmitting(false);
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
                    <div className="flex justify-center mb-6">
                        <img src="/img/new-logo-green.png" alt="Melbourne Open Data logo" className="h-16 w-auto" />
                    </div>

                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white text-center mb-1">
                        Welcome Back
                    </h1>
                    <p className="text-sm text-gray-500 dark:text-gray-400 text-center mb-8">
                        Sign in to your account
                    </p>

                    {error && (
                        <div className="rounded-lg bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 p-3 text-sm mb-5">
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit} noValidate className="space-y-5">
                        <div>
                            <label htmlFor="emailInput" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                                {t("Email")}
                            </label>
                            <input
                                type="email"
                                id="emailInput"
                                name="email"
                                placeholder="you@example.com"
                                value={email}
                                onChange={handleChange}
                                className="w-full px-4 py-3.5 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition"
                            />
                        </div>

                        <div>
                            <label htmlFor="passwordInput" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                                {t("Password")}
                            </label>
                            <div className="relative">
                                <input
                                    type={passwordVisible ? "text" : "password"}
                                    id="passwordInput"
                                    name="password"
                                    placeholder="••••••••"
                                    value={password}
                                    onChange={handleChange}
                                    className="w-full px-4 py-3.5 pr-12 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition"
                                />
                                <button
                                    type="button"
                                    onClick={togglePasswordVisibility}
                                    aria-label={passwordVisible ? "Hide password" : "Show password"}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                                >
                                    {passwordVisible ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                                </button>
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={isSubmitting}
                            className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-green-600 hover:bg-green-700 text-white font-semibold py-3.5 transition mt-6 disabled:cursor-not-allowed disabled:opacity-70"
                        >
                            {isSubmitting ? (
                                <>
                                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                                    Signing in...
                                </>
                            ) : (
                                "Sign In"
                            )}
                        </button>
                    </form>

                    <div className="mt-6 text-center space-y-2">
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                            Don&apos;t have an account?{" "}
                            <Link href="/signup" className="text-green-600 hover:text-green-700 font-medium">
                                Sign Up
                            </Link>
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                            <Link href="/forgot-password" className="text-green-600 hover:text-green-700 font-medium">
                                Forgot your password?
                            </Link>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default LoginForm;