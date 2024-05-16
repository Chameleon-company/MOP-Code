"use client";

import React, { useState } from "react";
import "../../../../public/styles/login.css";

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
    <div className="bg-white p-4 max-w-90vw w-auto box-border relative text-center mt-16">
      <div className="bg-white p-4 flex items-center justify-between w-full fixed top-0 left-0 box-border">
        <img src="/img/new-logo-green.png" alt="Chameleon Logo" className="logo h-16" />
        <div className="flex items-center">
          <span className="mr-4 text-sm text-gray-700">No Account yet?</span>
          <a href="/en/signup"><button className="px-2 py-1 bg-white text-gray-700 border border-gray-300 rounded cursor-pointer text-sm">Sign Up</button></a>
        </div>
      </div>
      <div className="login-container bg-white p-4 max-w-90vw w-auto box-border relative text-center mt-16 ">
        <h2>Account Log In</h2>
        <p className="subtitle">Please login to continue to your account</p>
        <form
          onSubmit={handleSubmit}
          action="/submit-your-login-form"
          method="POST"
        >
          <div className="relative flex items-center">
            <label htmlFor="email" className="sr-only px-4 py-2">Email</label>
            <input
              class="w-full px-4 py-2 border border-gray-300"
              type="email"
              id="email"
              name="email"
              required
              placeholder="Email"
              value={email}
              onChange={handleChange}
            />
          </div>
          <div className="relative flex items-center">
            <label htmlFor="password" className="sr-only px-4 py-2">Password</label>
            <input
              class="w-full px-4 py-2 border border-gray-300"
              type={passwordVisible ? "text" : "password"}
              id="password"
              name="password"
              required
              placeholder="Password"
              value={password}
              onChange={handleChange}
            />
            <span
              className="absolute right-4 cursor-pointer text-xl"
              onClick={togglePasswordVisibility}
              tabIndex="0" // Ensure element is focusable
            >
              {passwordVisible ? "🔒" : "🔓"}{" "}
              {/* Adjusted icons for visibility */}
            </span>
          </div>
          <div className="flex justify-between items-center w-full px-2">
            <label className="checkbox-label" htmlFor="remember-me">
              <input type="checkbox" id="remember-me" name="remember-me" />
              Remember Me
            </label>
            <a href="/en/forgot-password" className="px-4 py-2 text-blue-600 no-underline rounded">
              Forgot Password?
            </a>
          </div>
          <button type="submit" className="px-2 py-1 bg-green-500 text-white border border-gray-300 rounded cursor-pointer text-sm w-full">
            LOGIN
          </button>
        </form>
      </div>
      {error && <div className="error text-red-500">{error}</div>}
    </div>
  );
}

export default LoginForm;
