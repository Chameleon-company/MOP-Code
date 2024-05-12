'use client'

import React, { useState } from 'react';
import '../../../public/styles/login.css';

function LoginForm() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [passwordVisible, setPasswordVisible] = useState(false);
    const [error, setError] = useState('');

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;
        if (name === 'email') setEmail(value);
        else setPassword(value);
    };

    const togglePasswordVisibility = () => {
        setPasswordVisible(!passwordVisible);
    };

    const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!email || !password) {
            setError('Please fill in both fields');
            return;
        }
        console.log('Authentication in progress...'); 
        setError('');
    };

    return (
        <div className="login-container">
            <div className="top-bar">
                <img src="/img/image.png" alt="Chameleon Logo" className="logo"/>
                <div className="signup-container">
                    <span className="no-account">No Account yet?</span>
                    <button className="sign-up-button">Sign Up</button>
                </div>
            </div>
            <div className="login-container">
                <h2>Account Log In</h2>
                <p className="subtitle">Please login to continue to your account</p>
                <form onSubmit={handleSubmit} action="/submit-your-login-form" method="POST">
                    <div className="input-group">
                        <input type="email" id="email" name="email" required placeholder="Email"
                               value={email} onChange={handleChange} />
                    </div>
                    <div className="input-group">
                        <input type={passwordVisible ? "text" : "password"} id="password" name="password"
                               required placeholder="Password" value={password} onChange={handleChange} />
                        <span className="toggle-password" onClick={togglePasswordVisibility}>
                            {passwordVisible ? '🔒' : '🔓'} {/* Adjusted icons for visibility */}
                        </span>
                    </div>
                    <div className="options-container">
                        <label className="checkbox-label">
                            <input type="checkbox" id="remember-me" name="remember-me" />
                            Remember Me
                        </label>
                        <a href="/forgot-password" className="forgot-password">Forgot Password?</a>
                    </div>
                    <button type="submit" className="login-button">LOGIN</button>
                </form>
            </div>
            {error && <div className="error text-red-500">{error}</div>}
        </div>
    );
}

export default LoginForm;
