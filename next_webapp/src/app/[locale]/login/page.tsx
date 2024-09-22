'use client'

import React, { useState } from 'react';
import '../../../../public/styles/login.css';
import Header from "../../../components/Header";
import Link from 'next/link';
import { useTranslations } from "next-intl";
import Footer from "../../../components/Footer";


function LoginForm() {
    const t = useTranslations("login");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [passwordVisible, setPasswordVisible] = useState(false);
    const [error, setError] = useState("");

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;
        if (name === "email") setEmail(value);
        else setPassword(value);
    };

    const togglePasswordVisibility = () => {
        setPasswordVisible(!passwordVisible);
    };

    const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!email || !password) {
            setError("Please fill in both fields");
            return;
        }
        console.log("Authentication in progress...");
        setError("");
    };

    return (
        <>
            <div className="w-full fixed top-0 bg-white z-50">
                <Header />
            </div>
            <div className="main-content login-container">
                <div className="login-content mt-16"> {/* Adjusted margin-top for title */}
                    <h1 className="login-title">{t("Account Log In")}</h1>
                    <p className="login-subtitle">{t("Please login to continue to your account")}</p>
                    <form onSubmit={handleSubmit} action="/submit-your-login-form" method="POST">
                        <div className="mb-4">
                            <label htmlFor="emailInput" className="sr-only">
                                Email
                            </label>
                            <input
                                type="email"
                                id="emailInput"
                                placeholder={t("Email")}
                                className="w-full p-3 rounded-md border-solid border-2 border-[#ccc] bg-[#e9ebeb] login-input-wide" // Made wider
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
                                className="w-full p-3 rounded-md border-solid border-2 border-[#ccc] bg-[#e9ebeb] login-input-wide" // Made wider
                                value={password}
                                onChange={handleChange}
                                name="password"
                            />
                            <span className="absolute right-4 top-3 cursor-pointer" onClick={togglePasswordVisibility}>
                                {passwordVisible ? "👁️" : "👁️‍🗨️"} {/* Eye icon */}
                            </span>
                        </div>
                        <div className="options-container flex justify-between mb-4">
                            <label className="checkbox-label remember-me">
                                <input type="checkbox" id="remember-me" name="remember-me" />
                                {t("Remember Me")}
                            </label>
                            <Link href="forgot-password">
                                {t("Forgot Password?")}
                            </Link>
                        </div>
                        <button type="submit" className="login-button wide-button">{t("LOGIN")}</button> {/* Wider button */}
                    </form>
                    {error && <div className="error text-red-500 mt-4">{error}</div>}
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

