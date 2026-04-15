'use client'

import React, { useState } from 'react';
import '../../../../public/styles/login.css';
import Header from "../../../components/Header";
import { useTranslations, useLocale } from "next-intl";
import { useRouter } from 'next/navigation';
import Footer from "../../../components/Footer";

function LoginForm() {
    const t = useTranslations("login");
    const locale = useLocale();
    const router = useRouter();
    const [email, setEmail] = useState<string>("");
    const [password, setPassword] = useState<string>("");
    const [passwordVisible, setPasswordVisible] = useState(false);
    const [error, setError] = useState<string>("");

    const handleChange = (event) => {
        const { name, value } = event.target;
        if (name === "email") setEmail(value);
        else setPassword(value);
    };

    const togglePasswordVisibility = () => {
        setPasswordVisible(!passwordVisible);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!email || !password) {
            setError("Please fill in both fields");
            return;
        }

        try {
            const response = await fetch("/api/auth/login", {
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

            setError("");

            // Store user data in localStorage
            localStorage.setItem("userId", result.data.userId.toString());
            localStorage.setItem("user", JSON.stringify(result.data));
            localStorage.setItem("token", result.data.token);

            // Redirect based on role
            if (result.data.roleId === 1) {
                router.push(`/${locale}/admin/dashboard`);
            } else {
                router.push(`/${locale}/profile`);
            }

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
            <div className="main-content login-container dark:bg-[#263238]">
                <div className="login-content mt-16">
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
                                placeholder={t("Email")}
                                className="w-full p-3 rounded-md border-solid border-2 border-[#ccc] bg-[#e9ebeb] login-input-wide"
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
                                className="w-full p-3 rounded-md border-solid border-2 border-[#ccc] bg-[#e9ebeb] login-input-wide"
                                value={password}
                                onChange={handleChange}
                                name="password"
                            />
                            <span className="absolute right-4 top-3 cursor-pointer" onClick={togglePasswordVisibility}>
                                {passwordVisible ? "👁️" : "👁️‍🗨️"}
                            </span>
                        </div>
                        <div className="options-container flex justify-between mb-4">
                            <label className="checkbox-label remember-me dark:text-[#FFFFFF]">
                                <input type="checkbox" id="remember-me" name="remember-me" />
                                {t("Remember Me")}
                            </label>
                            <a href="#" className="forgot-password dark:text-[#FFFFFF]">{t("Forgot Password?")}</a>
                        </div>
                        <button type="submit" className="login-button wide-button">{t("LOGIN")}</button>
                    </form>
                    {error && <div className="error text-red-500 mt-4">{error}</div>}
                    <p className="text-center mt-4 dark:text-white">
                        Don&apos;t have an account?{" "}
                        <a href={`/${locale}/signup`} className="text-[#2DBE6C] font-semibold hover:underline">
                            Sign Up
                        </a>
                    </p>
                </div>
            </div>
            {/* Logo */}
            <div className="absolute top-20 left-20 mr-4 m-4">
                <img src="/img/new-logo-green.png" alt="Logo" className="h-40" />
            </div>
            <Footer />
        </>
    );
}

export default LoginForm;