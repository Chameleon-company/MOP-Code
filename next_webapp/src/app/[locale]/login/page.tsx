'use client'

import React, { useState } from 'react';
import '../../../../public/styles/login.css';
import Header from "../../../components/Header";
import { useTranslations } from "next-intl";

function LoginForm() {
    const t = useTranslations("login");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [passwordVisible, setPasswordVisible] = useState(false);
    const [error, setError] = useState("");

    const handleChange = (event) => {
        const { name, value } = event.target;
        if (name === "email") setEmail(value);
        else setPassword(value);
    };

    const togglePasswordVisibility = () => {
        setPasswordVisible(!passwordVisible);
    };

    const handleSubmit = (event) => {
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
            <div className="login-container">
                <div className="top-bar">
                    <img src="/img/image.png" alt="Chameleon Logo" className="logo" />
                </div>
                <div className="login-content mt-20">
                    <h1 className="login-title">{t("Account Log In")}</h1>
                    <p className="login-subtitle">{t("Please login to continue to your account")}</p>
                    <form onSubmit={handleSubmit} action="/submit-your-login-form" method="POST">
                        <div className="input-group">
                            <input
                                type="email"
                                id="email"
                                name="email"
                                required
                                placeholder={t("Email")}
                                value={email}
                                onChange={handleChange}
                            />
                        </div>
                        <div className="input-group">
                            <input
                                type={passwordVisible ? "text" : "password"}
                                id="password"
                                name="password"
                                required
                                placeholder={t("Password")}
                                value={password}
                                onChange={handleChange}
                            />
                            <span className="toggle-password" onClick={togglePasswordVisibility}>
                                {passwordVisible ? "🔒" : "🔓"}
                            </span>
                        </div>
                        <div className="options-container">
                            <label className="checkbox-label remember-me">
                                <input type="checkbox" id="remember-me" name="remember-me" />
                                {t("Remember Me")}
                            </label>
                            <a href="#" className="forgot-password">{t("Forgot Password?")}</a>
                        </div>
                        <button type="submit" className="login-button">{t("LOGIN")}</button>
                    </form>
                    {error && <div className="error text-red-500">{error}</div>}
                </div>
            </div>
        </>
    );
}

export default LoginForm;