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
    <div className=" flex flex-col min-h-screen">
      <Header/>
      <div className="signup-page flex flex-col items-center justify-center flex-grow" style={{ backgroundColor: "#fffbfb"}}>
        <div className="p-8 rounded-lg mt-12">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold mb-12">ACCOUNT SIGN UP</h2>
          </div>
          <div className="mb-4 flex">
            <label htmlFor="firstNameInput" className="sr-only">
              First Name
            </label>
            <input
              type="text"
              id="firstNameInput"
              placeholder= "First Name"
              className="w-full p-3 border border-gray-300 bg-white mr-2"
            />
            <label htmlFor="lastNameInput" className="sr-only">
              Last Name
            </label>
            <input
              type="text"
              id="lastNameInput"
              placeholder="Last Name"
              className="w-full p-3 border border-gray-300 bg-white mr-2"
            />
          </div>
          <div className="mb-4">
            <label htmlFor="emailInput" className="sr-only">
              Email
            </label>
            <input
              type="email"
              id="emailInput"
              placeholder="Email"
              className="w-full p-3 border border-gray-300 bg-white"
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
              className="w-full p-3 border border-gray-300 bg-white"
              onChange={handlePasswordChange}
            />
          </div>
          <div className="mb-4">
            <div className="w-full h-2 border border-gray-300 rounded mb-2">
              <div className="h-full rounded" style={passwordStrengthStyle}></div>
            </div>
            <p className="text-right text-sm text-gray-600 mt-2">
             {passwordStrength}
            </p>
          </div>
          <button className="w-full bg-green-500 font-bold text-white py-2 px-2 cursor-pointer">
            NEXT
          </button>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default SignUpPage;
