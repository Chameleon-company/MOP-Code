"use client";

import React, { useState } from "react";
import Head from "next/head";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { motion } from "framer-motion";

const SignUpPage = () => {
  const t = (key: string) => key;
  const [form, setForm] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
  });
  const [errors, setErrors] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
  });
  const [passwordStrength, setPasswordStrength] = useState<string>("");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setForm({ ...form, [name]: value });
    if (name === "password") setPasswordStrength(checkPasswordStrength(value));
  };

  const checkPasswordStrength = (password: string): string => {
    const hasUpper = /[A-Z]/.test(password);
    const hasLower = /[a-z]/.test(password);
    const hasNumber = /[0-9]/.test(password);
    const hasSpecial = /[!@#$%^&*]/.test(password);
    const lengthValid = password.length >= 8;

    if (!lengthValid) return t("Weak");
    const passed = [hasUpper, hasLower, hasNumber, hasSpecial].filter(
      Boolean
    ).length;
    if (passed >= 3) return t("Strong");
    if (passed === 2) return t("Moderate");
    return t("Weak");
  };

  const getPasswordStrengthColor = () => {
    switch (passwordStrength) {
      case t("Weak"):
        return "#e74c3c";
      case t("Moderate"):
        return "#f39c12";
      case t("Strong"):
        return "#2ECC71";
      default:
        return "transparent";
    }
  };

  const validateForm = () => {
    const newErrors = {
      firstName: form.firstName ? "" : t("Please enter your first name."),
      lastName: form.lastName ? "" : t("Please enter your last name."),
      email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)
        ? ""
        : t("Please enter a valid email address."),
      password: "",
    };

    const { password } = form;
    const hasUpper = /[A-Z]/.test(password);
    const hasLower = /[a-z]/.test(password);
    const hasNumber = /[0-9]/.test(password);
    const hasSpecial = /[!@#$%^&*]/.test(password);
    const lengthValid = password.length >= 8;

    if (!lengthValid) {
      newErrors.password = t(
        "Your password must be at least 8 characters long."
      );
    } else if (!hasUpper || !hasLower || !hasNumber || !hasSpecial) {
      newErrors.password = t(
        "Password must include at least one uppercase letter, one lowercase letter, one number, and one special character."
      );
    }

    setErrors(newErrors);
    return Object.values(newErrors).every((err) => err === "");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateForm()) return;

    try {
      const response = await fetch("/api/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await response.json();
      if (response.ok) {
        alert("Sign-up successful!");
      } else {
        alert(`Sign-up failed: ${data.message}`);
      }
    } catch (err) {
      alert("Something went wrong. Please try again later.");
    }
  };

  return (
    <>
      <Head>
        <link
          href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap"
          rel="stylesheet"
        />
      </Head>
      <div
        className="signup-page flex flex-col min-h-screen bg-gradient-to-b from-[#2ECC71] to-white dark:bg-gradient-to-b dark:from-[#263238] dark:to-[#263238] text-black font-semibold "
        style={{ fontFamily: "Poppins, sans-serif" }}
      >
        <Header showSignUpButton={false} />
        <div className="min-h-[100vh] flex flex-col items-center justify-center px-4 py-10 bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 dark:from-white dark:via-gray-100 dark:to-white transition-colors duration-500">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-[500px]"
          >
            <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl p-8 shadow-xl transition-colors">
              <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent text-center mb-2">
                {t("Account Sign up")}
              </h2>
              <p className="text-gray-600 dark:text-gray-300 text-center mb-8">
                Register your details below to create your account
              </p>

              <form onSubmit={handleSubmit} noValidate className="space-y-6">
                <div className="flex gap-4">
                  <div className="w-1/2">
                    <input
                      name="firstName"
                      type="text"
                      placeholder={t("First name")}
                      value={form.firstName}
                      onChange={handleChange}
                      className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-800 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 rounded-lg p-3 focus:outline-none focus:border-blue-500 transition-colors"
                    />
                    {errors.firstName && (
                      <p className="text-red-500 text-xs mt-1">
                        {errors.firstName}
                      </p>
                    )}
                  </div>
                  <div className="w-1/2">
                    <input
                      name="lastName"
                      type="text"
                      placeholder={t("Last name")}
                      value={form.lastName}
                      onChange={handleChange}
                      className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-800 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 rounded-lg p-3 focus:outline-none focus:border-blue-500 transition-colors"
                    />
                    {errors.lastName && (
                      <p className="text-red-500 text-xs mt-1">
                        {errors.lastName}
                      </p>
                    )}
                  </div>
                </div>

                <div>
                  <input
                    name="email"
                    type="email"
                    placeholder={t("Email address")}
                    value={form.email}
                    onChange={handleChange}
                    className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-800 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 rounded-lg p-3 focus:outline-none focus:border-blue-500 transition-colors"
                  />
                  {errors.email && (
                    <p className="text-red-500 text-xs mt-1">{errors.email}</p>
                  )}
                </div>

                <div>
                  <input
                    name="password"
                    type="password"
                    placeholder={t("Password")}
                    value={form.password}
                    onChange={handleChange}
                    className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-800 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 rounded-lg p-3 focus:outline-none focus:border-blue-500 transition-colors"
                  />
                  {errors.password && (
                    <p className="text-red-500 text-xs mt-1">
                      {errors.password}
                    </p>
                  )}
                </div>

                <div>
                  <div className="w-full h-2 rounded bg-gray-200 dark:bg-gray-700 overflow-hidden">
                    <div
                      className="h-full rounded transition-all"
                      style={{
                        width: `${Math.min(
                          (form.password.length / 10) * 100,
                          100
                        )}%`,
                        backgroundColor: getPasswordStrengthColor(),
                      }}
                    />
                  </div>
                  <p className="text-right text-sm mt-1 italic text-gray-600 dark:text-gray-300">
                    {t("Password Strength")}: {passwordStrength}
                  </p>
                </div>

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  type="submit"
                  className="w-full bg-green-500 text-white font-medium py-3 rounded-lg flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
                >
                  Sign Up
                </motion.button>
              </form>
            </div>
          </motion.div>
        </div>

        <Footer />
      </div>
    </>
  );
};

export default SignUpPage;
