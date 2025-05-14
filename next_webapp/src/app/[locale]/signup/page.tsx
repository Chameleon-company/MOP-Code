"use client";

import React, { useEffect, useState } from "react";
import Head from "next/head";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";

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
  const[darkMode,setDarkMode]=useState(false)
    useEffect(() => {
        const htmlElement = document.documentElement;
        const hasDarkClass = htmlElement.classList.contains("dark");
        setDarkMode(hasDarkClass);
    }, []);

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
    const passed = [hasUpper, hasLower, hasNumber, hasSpecial].filter(Boolean).length;
    if (passed >= 3) return t("Strong");
    if (passed === 2) return t("Moderate");
    return t("Weak");
  };

  const getPasswordStrengthColor = () => {
    switch (passwordStrength) {
      case t("Weak"): return "#e74c3c";
      case t("Moderate"): return "#f39c12";
      case t("Strong"): return "#2ECC71";
      default: return "transparent";
    }
  };

  const validateForm = () => {
    const newErrors = {
      firstName: form.firstName ? "" : t("Please enter your first name."),
      lastName: form.lastName ? "" : t("Please enter your last name."),
      email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email) ? "" : t("Please enter a valid email address."),
      password: "",
    };

    const { password } = form;
    const hasUpper = /[A-Z]/.test(password);
    const hasLower = /[a-z]/.test(password);
    const hasNumber = /[0-9]/.test(password);
    const hasSpecial = /[!@#$%^&*]/.test(password);
    const lengthValid = password.length >= 8;

    if (!lengthValid) {
      newErrors.password = t("Your password must be at least 8 characters long.");
    } else if (!hasUpper || !hasLower || !hasNumber || !hasSpecial) {
      newErrors.password = t("Password must include at least one uppercase letter, one lowercase letter, one number, and one special character.");
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
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet" />
      </Head>
      <div className="signup-page flex flex-col min-h-screen bg-gradient-to-b from-[#2ECC71] to-white text-black font-semibold dark:bg-gray-900 dark:text-white" style={{ fontFamily: 'Poppins, sans-serif' }}>
        <Header showSignUpButton={false} />
        <div className="flex flex-col items-center justify-center flex-grow px-4 dark:text-white dark:bg-gray-900">
          <div className="w-full max-w-xl p-4">
            <h2 className="text-4xl font-bold text-center mb-8">{t("Account Sign up")}</h2>
            <form onSubmit={handleSubmit} noValidate className="space-y-6">
              <div className="flex gap-4">
                <div className="w-1/2 dark:text-white">
                  <input
                    name="firstName"
                    type="text"
                    placeholder={t("First name")}
                    value={form.firstName}
                    onChange={handleChange}
                    className="w-full p-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#2ECC71] transition transform duration-200 ease-in-out focus:scale-105 hover:shadow-lg"
                  />
                  {errors.firstName && <p className="text-red-500 text-xs mt-1">{errors.firstName}</p>}
                </div>
                <div className="w-1/2 dark:text-white">
                  <input
                    name="lastName"
                    type="text"
                    placeholder={t("Last name")}
                    value={form.lastName}
                    onChange={handleChange}
                    className="w-full p-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#2ECC71] transition transform duration-200 ease-in-out focus:scale-105 hover:shadow-lg"
                  />
                  {errors.lastName && <p className="text-red-500 text-xs mt-1">{errors.lastName}</p>}
                </div>
              </div>
              <div>
                <input
                  name="email"
                  type="email"
                  placeholder={t("Email address")}
                  value={form.email}
                  onChange={handleChange}
                  className="w-full p-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#2ECC71] dark:text-white"
                />
                {errors.email && <p className="text-red-500 text-xs mt-1">{errors.email}</p>}
              </div>
              <div>
                <input
                  name="password"
                  type="password"
                  placeholder={t("Password")}
                  value={form.password}
                  onChange={handleChange}
                  className="w-full p-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#2ECC71]"
                />
                {errors.password && <p className="text-red-500 text-xs mt-1">{errors.password}</p>}
              </div>
              <div>
                <div className="w-full h-2 rounded bg-gray-200">
                  <div
                    className="h-full rounded transition-all"
                    style={{ width: `${Math.min((form.password.length / 10) * 100, 100)}%`, backgroundColor: getPasswordStrengthColor() }}
                  ></div>
                </div>
                <p className="text-right text-sm mt-1 italic text-gray-600 dark:text-white">{t("Password Strength")}: {passwordStrength}</p>
              </div>
              <button
                type="submit"
                className="w-full bg-white text-black text-lg font-semibold py-3 px-4 rounded-lg border border-[#2ECC71] hover:bg-[#2ECC71] hover:text-white transition duration-200"
              >
                {t("Next")}
              </button>
            </form>
          </div>
        </div>
        <Footer />
      </div>
    </>
  );
};

export default SignUpPage;
