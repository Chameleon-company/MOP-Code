// ForgotPasswordPage.js
"use client";
import React, { useState } from "react";
import Header from "../../../components/Header";
import "../../../../public/styles/forgot-password.css";
import { useTranslations } from "next-intl";
import Tooglebutton from "../Tooglebutton/Tooglebutton";
import Link from "next/link"; // Import Link from Next.js

const ForgotPasswordPage = () => {
  const t = useTranslations("forgot-password");
  //dark theme
  const [dark_value, setdarkvalue] = useState(false);
  const handleValueChange = (
    newValue: boolean | ((prevState: boolean) => boolean)
  ) => {
    setdarkvalue(newValue);
  };
  return (
    <div className={`${dark_value && "dark"}`}>
      <div className="bg-slate-50 dark:bg-[#1d1919] min-h-screen">
        {/* Header */}
        <div className="w-full">
          <Header />
        </div>
        <Tooglebutton onValueChange={handleValueChange} />
        <div className="flex flex-col items-center justify-center">
          {/* Forgot password form */}
          <div className="p-4 md:p-8 fpwd-form mt-16">
            <div className="text-center text-black dark:text-slate-200">
              <h2 className="text-xl md:text-2xl xl:text-3xl font-bold mt-4 pb-4">
                {t("Forgot Password")}
              </h2>
            </div>
            {/* Message for OTP */}
            <p className="text-black dark:text-slate-200 mb-14 text-[10px] md:text-[12px] xl:text-[16px] text-gray-600 font-bold text-wrap text-center">
              <span className="block">
                {t("line1")}
              </span>
              <span className="block">{t("line2")}</span>
            </p>
            <div className="mb-0">
              {/* Use htmlFor to associate label with input for screen readers */}
              <label htmlFor="email" className="sr-only">
                Email
              </label>
              <input
                type="email"
                id="email"
                name="email"
                placeholder={t("Email")}
                className="w-full p-2 rounded border-2 border-gray-300 dark:border-white mb-4 bg-gray-200 dark:bg-[#1d1919]"
              />
            </div>
            {/* Use Link from Next.js for navigation */}
            <Link href="/en/otp_verification">
              <button className="w-full bg-green-500 text-white p-2 rounded cursor-pointer hover:bg-green-600">
                {t("Continue")}
              </button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForgotPasswordPage;
