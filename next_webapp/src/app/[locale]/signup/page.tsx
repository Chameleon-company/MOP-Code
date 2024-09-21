"use client";

import React, { useState } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";

//Signup
const SignUpPage = () => {
  const t = useTranslations("signup");
  const [firstName, setFirstName] = useState<string>("");
  const [lastName, setLastName] = useState<string>("");
  const [email, setEmail] = useState<string>("");
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

  // Handle form submission
  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault(); // Prevent default form submission behavior

    console.log('Submitting form with:', { firstName, lastName, email, password });

    try {
      const response = await fetch('/api/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ firstName, lastName, email, password }),
      });

      const result = await response.json();
      console.log('Response from API:', result);

      if (response.ok) {
        alert("Sign-up successful!");
        // Optionally, redirect the user or clear the form
      } else {
        alert(`Sign-up failed: ${result.message}`);
      }
    } catch (error) {
      console.error('Error:', error);
      alert("An error occurred. Please try again.");
    }
  };

  return (
    <div className="signup-page flex flex-col min-h-screen">
      <Header showSignUpButton={false} />
      <div className="flex flex-col items-center justify-center flex-grow">
        <div className="p-8 rounded-lg mt-12">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold mb-12">{t("Account Sign Up")}</h2>
          </div>
          <form onSubmit={handleSubmit}>
            <div className="mb-4 flex">
              <label htmlFor="firstNameInput" className="sr-only">
                First Name
              </label>
              <input
                type="text"
                id="firstNameInput"
                placeholder={t("First name")}
                className="w-1/2 p-2 rounded-md border-solid border-2 border-[#ccc] mr-2 bg-[#e9ebeb]"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
              />
              <label htmlFor="lastNameInput" className="sr-only">
                Last Name
              </label>
              <input
                type="text"
                id="lastNameInput"
                placeholder={t("Last name")}
                className="w-1/2 p-2 rounded-md border-solid border-2 border-[#ccc] mr-2 bg-[#e9ebeb]"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
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
                value={email}
                onChange={(e) => setEmail(e.target.value)}
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
                value={password}
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
            <button
              type="submit"
              className="w-full bg-green-500 text-white py-2 px-2 rounded-md cursor-pointer"
            >
              {t("Next")}
            </button>
          </form>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default SignUpPage;