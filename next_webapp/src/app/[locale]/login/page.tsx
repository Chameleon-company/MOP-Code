'use client'

import React, { useState } from 'react';
import '../../../../public/styles/login.css';

function LoginForm() {
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
        <div className="login-container">
            <div className="top-bar">
                <img src="/img/image.png" alt="Chameleon Logo" className="logo" />
                <div className="signup-container">
                    <span className="no-account">No Account yet?</span>
                    <a href="./signup" className="sign-up-button">Sign Up</a>
                </div>
            </div>
            <div className="login-content">
                <h1 className="login-title">Account Log In</h1>
                <p className="login-subtitle">Please login to continue to your account</p>
                <form onSubmit={handleSubmit} action="/submit-your-login-form" method="POST">
                    <div className="input-group">
                        <input
                            type="email"
                            id="email"
                            name="email"
                            required
                            placeholder="Email"
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
                            placeholder="Password"
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
                            Remember Me
                        </label>
                        <a href="#" className="forgot-password">Forgot Password?</a>
                    </div>
                    <button type="submit" className="login-button">LOGIN</button>
                </form>
                {error && <div className="error text-red-500">{error}</div>}
            </div>
        </div>
    );
}

export default LoginForm;