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
            setError("PLEASE ENTER YOUR EMAIL AND PASSWORD");
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

            const response = await fetch("/api/login", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ email, password }),
            });

            const result = await response.json();
            console.log("Login API response:", result);

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

            setError("");
            alert("Login successful!");

            // Optionally: Store user data in localStorage/sessionStorage
            localStorage.setItem("user", JSON.stringify(result.user));

            // Redirect to home page (or dashboard)
            router.push("/"); // Update to your actual home route

        } catch (error) {
            console.error("Login error:", error);
            setError("Something went wrong. Please try again.");
        }
    };

    return (
        <>
            <div className="w-full fixed top-0 bg-white dark:bg-[#263238] z-50">
                <Header />
            </div>
            <div className="main-content login-container dark:bg-[#263238] z-10">
                <div className="login-content mt-16"> {/* Adjusted margin-top for title */}
                    <h1 className="login-title dark:text-[#FFFFFF]">{t("Account Log In")}</h1>
                    <p className="login-subtitle dark:text-[#FFFFFF]">{t("Please login to continue to your account")}</p>
                    <form onSubmit={handleSubmit} action="/submit-your-login-form" method="POST">
                        <div className="mb-4">
                            <label htmlFor="emailInput" className="sr-only">
                                Email
                            </label>
                            <input
                                type="email"
                                id="emailInput"
                                name="email"
                                placeholder="you@example.com"
                                value={email}
                                placeholder={t("Email")}
                                className="w-full p-3 rounded-md border-solid border-2 border-gray-600 bg-[#e9ebeb] login-input-wide" // Made wider
                                value={email}
                                onChange={handleChange}
                                name="email"
                            />
                        </div>
                        <div className="mb-4 relative">
                            <label htmlFor="passwordInput" className="sr-only">
                                Password
                            </label>
                            <input
                                type={passwordVisible ? "text" : "password"}
                                id="passwordInput"
                                placeholder={t("Password")}
                                className="w-full p-3 rounded-md border-solid border-2 border-gray-600 bg-[#e9ebeb] login-input-wide" // Made wider
                                value={password}
                                onChange={handleChange}
                                className="w-full px-4 py-3.5 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 transition"
                            />
                        </div>

                        <div>
                            <label htmlFor="passwordInput" className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                                {t("Password")}
                        <button type="submit" className="login-button border-gray-600 wide-button">{t("LOGIN")}</button> {/* Wider button */}
                        <div className="options-container flex justify-between mb-4 pt-4">
                            <label className="checkbox-label remember-me dark:text-[#FFFFFF]">
                                <input type="checkbox" id="remember-me" name="remember-me" />
                                {t("Remember Me")}
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
                        <div className="auth-separator">
                            <span>or continue with</span>
                        </div>

                        <div className="social-auth">
                            <button
                                type="button"
                                className="social-btn"
                                onClick={() => handle('google')}
                                aria-label="Continue with Google"
                            >
                                <img src="/img/google.svg" alt="" aria-hidden="true" />
                                <span>Google</span>
                            </button>

                            <button
                                type="button"
                                className="social-btn"
                                onClick={() => handle('apple')}
                                aria-label="Continue with Apple"
                            >
                                <img src="/img/apple.svg" alt="" aria-hidden="true" />
                                <span>Apple</span>
                            </button>

                            <button
                                type="button"
                                className="social-btn"
                                onClick={() => handle('facebook')}
                                aria-label="Continue with Facebook"
                            >
                                <img src="/img/facebook.svg" alt="" aria-hidden="true" />
                                <span>Facebook</span>
                            </button>
                        </div>
                    </form>
                    {error && <div className="error text-red-500 mt-4 items-center justify-center">{error}</div>}
                </div>
            </div>
            {/* Logo */}
            <div className="absolute inset-0 flex z-0 items-center justify-center">
                <img
                    src="/img/new-logo-green.png"
                    alt="Logo"
                    className="w-full h-full object-contain opacity-40"
                />
            </div>
            <Footer />
        </>
    );
}

export default LoginForm;