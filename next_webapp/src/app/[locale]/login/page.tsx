'use client'

import React, { useState } from 'react';
import '../../../../public/styles/login.css';
import Header from "../../../components/Header";
import { useTranslations } from "next-intl";
import { useRouter } from 'next/navigation';
import Footer from "../../../components/Footer";

function LoginForm() {
    const t = useTranslations("login");
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
            setError("PLEASE ENTER YOUR EMAIL AND PASSWORD");
            return;
        }

        try {
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
                                name="password"
                            />
                            <span className="absolute right-4 top-3 cursor-pointer" onClick={togglePasswordVisibility}>
                                {passwordVisible ? "👁️" : "👁️‍🗨️"} {/* Eye icon */}
                            </span>
                        </div>
                        <button type="submit" className="login-button border-gray-600 wide-button">{t("LOGIN")}</button> {/* Wider button */}
                        <div className="options-container flex justify-between mb-4 pt-4">
                            <label className="checkbox-label remember-me dark:text-[#FFFFFF]">
                                <input type="checkbox" id="remember-me" name="remember-me" />
                                {t("Remember Me")}
                            </label>
                            <a href="#" className="forgot-password dark:text-[#FFFFFF]">{t("Forgot Password?")}</a>
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
