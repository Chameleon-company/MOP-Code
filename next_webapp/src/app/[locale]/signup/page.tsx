"use client";

import React, { useState } from "react";
import Header from "@/components/header";
import Footer from "@/components/footer";

const SignUpPage = () => {
  const [password, setPassword] = useState("");
  const [passwordStrength, setPasswordStrength] = useState("");

  const handlePasswordChange = (event) => {
    const newPassword = event.target.value;
    setPassword(newPassword);
    setPasswordStrength(checkPasswordStrength(newPassword));
  };

  const checkPasswordStrength = (password) => {
    if (password.length < 6) {
      return "Weak";
    } else if (password.length < 10) {
      return "Moderate";
    } else {
      return "Strong";
    }
  };

  const getPasswordStrengthColor = () => {
    switch (passwordStrength) {
      case "Weak":
        return "red";
      case "Moderate":
        return "orange";
      case "Strong":
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
            <h2 className="text-2xl font-bold mb-12">Account Sign Up</h2>
          </div>
          <div className="mb-4 flex">
            <label htmlFor="firstNameInput" className="sr-only">First Name</label>
            <input
              type="text"
              id="firstNameInput"
              placeholder="First name"
              className="w-1/2 p-2 rounded-md border-solid border-2 border-[#ccc] mr-2 bg-[#e9ebeb]"
            />
            <label htmlFor="lastNameInput" className="sr-only">Last Name</label>
            <input
              type="text"
              id="lastNameInput"
              placeholder="Last name"
              className="w-1/2 p-2 rounded-md border-solid border-2 border-[#ccc] mr-2 bg-[#e9ebeb]"
            />
          </div>
          <div className="mb-4">
            <label htmlFor="emailInput" className="sr-only">Email</label>
            <input
              type="email"
              id="emailInput"
              placeholder="Email"
              className="w-full p-2 rounded-md border-solid border-2 border-[#ccc] mr-2 bg-[#e9ebeb]"
            />
          </div>
          <div className="mb-4">
            <label htmlFor="passwordInput" className="sr-only">Password</label>
            <input
              type="password"
              id="passwordInput"
              placeholder="Password"
              className="w-full p-2 rounded-md border-solid border-2 border-[#ccc] mr-2 bg-[#e9ebeb]"
              onChange={handlePasswordChange}
            />
          </div>
          <div className="mb-4">
            <div className="w-full h-2 border border-gray-300 rounded mb-2">
              <div className="h-full rounded" style={passwordStrengthStyle}></div>
            </div>
            <p className="text-right text-sm text-gray-600 mt-2">
              Password Strength: {passwordStrength}
            </p>
          </div>
          <button className="w-full bg-green-500 text-white py-2 px-2 rounded-md cursor-pointer">
            Next
          </button>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default SignUpPage;



