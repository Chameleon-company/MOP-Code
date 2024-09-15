// ForgotPasswordPage.js

import React from "react";
import Link from "next/link"; // Import Link from Next.js
import Header from "../../../components/Header";

const ForgotPasswordPage = () => {
  return (
    
      <div className="bg-white w-full dark:bg-zinc-800 text-black dark:text-slate-100 min-h-screen">
        <Header />

        <div className="flex flex-col items-center justify-center h-[80vh]">
          {/* Forgot password form */}
          <div className="rounded-lg w-3/4 md:w-2/4 xl:w-1/4">
            <div className="text-center">
              <h2 className="text-2xl font-bold mt-4 pb-4">Forgot Password</h2>
            </div>
            {/* Message for OTP */}
            <p className="mb-14 text-sm text-gray-600 dark:text-zinc-300 font-bold text-center">
              <span className="block">ONE-TIME PASSWORD (OTP) WILL BE SENT TO YOUR</span>
              <span className="block">EMAIL ADDRESS FOR VERIFICATION</span>
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
                placeholder="Email"
                className="w-full p-2 rounded border border-gray-300 dark:bg-zinc-800 mb-4 bg-gray-200"
              />
            </div>
            {/* Use Link from Next.js for navigation */}
            <Link href="/en/otp_verification">
              <button className="w-full bg-green-500 text-white p-2 rounded cursor-pointer hover:bg-green-600">Continue</button>
            </Link>
          </div>

        </div>
      </div>
  

  );
};

export default ForgotPasswordPage;
