"use client";

import React, { useState } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import LanguageDropdown from "../../../components/LanguageDropdown";
import { useTranslations } from "next-intl";

const SignUpPage = () => {
  const t = useTranslations("signup");
  const [password, setPassword] = useState<string>("");
  const [passwordStrength, setPasswordStrength] = useState<string>("");

  const handlePasswordChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newPassword = event.target.value;
    setPassword(newPassword);
    setPasswordStrength(checkPasswordStrength(newPassword));
  };

  const checkPasswordStrength = (password: string): string => {
    if (password.length < 6) {
      return t("Weak");
    } else if (password.length < 10) {
      return t("Moderate");
    } else {
      return t("Strong");
    }
  };

  const getPasswordStrengthColor = (): string => {
    switch (passwordStrength) {
      case t("Weak"):
        return "red";
      case t("Moderate"):
        return "orange";
      case t("Strong"):
        return "green";
      default:
        return "transparent";
    }
  };

  const passwordStrengthStyle = {
    width: `${Math.min((password.length / 10) * 100, 100)}%`,
    backgroundColor: getPasswordStrengthColor(),
  };

  return (
    <div className="signup-page flex flex-col min-h-screen">
      <Header showSignUpButton={false} />
      <div className="flex flex-col items-center justify-center flex-grow">
        <div className="p-8 rounded-lg mt-12">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold mb-12">{t("Account Sign Up")}</h2>
          </div>
          <div className="mb-4 flex">
            <label htmlFor="firstNameInput" className="sr-only">
              First Name
            </label>
            <input
              type="text"
              id="firstNameInput"
              placeholder={t("First name")}
              className="w-1/2 p-2 rounded-md border-solid border-2 border-[#ccc] mr-2 bg-[#e9ebeb]"
            />
            <label htmlFor="lastNameInput" className="sr-only">
              Last Name
            </label>
            <input
              type="text"
              id="lastNameInput"
              placeholder={t("Last name")}
              className="w-1/2 p-2 rounded-md border-solid border-2 border-[#ccc] mr-2 bg-[#e9ebeb]"
            />
          </div>
          <div className="mb-4">
            <label htmlFor="emailInput" className="sr-only">
              Email
            </label>
            <input
              type="email"
              id="emailInput"
              placeholder={t("Email")}
              className="w-full p-2 rounded-md border-solid border-2 border-[#ccc] bg-[#e9ebeb]"
            />
          </div>
          <div className="mb-4">
            <label htmlFor="passwordInput" className="sr-only">
              Password
            </label>
            <input
              type="password"
              id="passwordInput"
              placeholder={t("Password")}
              className="w-full p-2 rounded-md border-solid border-2 border-[#ccc] bg-[#e9ebeb]"
              onChange={handlePasswordChange}
            />
          </div>
          <div className="mb-4">
            <div className="w-full h-2 border border-gray-300 rounded mb-2">
              <div className="h-full rounded" style={passwordStrengthStyle}></div>
            </div>
            <p className="text-right text-sm text-gray-600 mt-2">
              {t("Password Strength")}: {passwordStrength}
            </p>
          </div>
          <button className="w-full bg-green-500 text-white py-2 px-2 rounded-md cursor-pointer">
            {t("Next")}
          </button>
        </div>
      </div>

      {/* Already a member section */}
      <div className="absolute top-20 right-20 ml-4 m-4 flex items-center text-[#666666]">
        <p className="text-sm mr-2">{t("Already a member?")}</p>
        <a
          href="/en/login"
          className="border border-gray-600 text-gray-600 px-4 py-2 ml-4 hover:border-[#999999]"
        >
          {t("Log In")}
        </a>
        <div className="ml-4">
          <LanguageDropdown />
        </div>
      </div>

      {/* Logo */}
      <div className="absolute top-20 left-20 mr-4 m-4">
        <img src="/img/new-logo-green.png" alt="Logo" className="h-40" />
      </div>
      <Footer />
    </div>
  );
};

export default SignUpPage;
