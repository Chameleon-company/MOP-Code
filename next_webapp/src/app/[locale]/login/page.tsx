'use client'; 

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';  // Import from 'next/navigation' for Next.js 13+ routing
import Header from "../../../components/Header";  // Import the header component
import '../../../../public/styles/login.css';

function LoginForm() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [passwordVisible, setPasswordVisible] = useState(false);
    const [error, setError] = useState("");
    const router = useRouter(); // Router for navigation (Next.js 13+)

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

        // Simulate successful authentication
        setTimeout(() => {
            router.push('/home');  // Redirect to home screen
        }, 1000);

        setError("");
    };

    return (
        <div>
            <Header /> {/* Include the Header */}
            <div className="wrapper">
            <div className="login-container">
                <div className="login-content">
                    <h1 className="login-title">Account Log In</h1>
                    <p className="login-subtitle">Please login to continue to your account</p>
                    <form onSubmit={handleSubmit} method="POST">
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
                                {passwordVisible ? "👁️" : "👁️‍🗨️"}
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
            </div>
        </div>
    );
}

export default LoginForm;