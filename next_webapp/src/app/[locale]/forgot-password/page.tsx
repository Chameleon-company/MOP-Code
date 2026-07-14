// ForgotPasswordPage.js

import React from "react";
import Link from "next/link"; // Import Link from Next.js

const ForgotPasswordPage = () => {
  return (
    <div className="bg-white text-black flex flex-col items-center justify-center min-h-screen">
      {/* No Account Yet */}
      <div className="absolute top-20 right-20 m-4 flex items-center text-gray-600">
        <p className="text-sm mr-2">No Account Yet?</p>
        {/* Use Link from Next.js for navigation */}
        <Link href="/en/signup">
          <p className="border border-black p-2 text-base text-black">Sign up</p>
        </Link>
      </div>

      {/* Logo */}
      <div className="absolute top-20 left-20 m-4">
        {/* Add your logo here */}
        <img src="/img/new-logo-green.png" alt="Logo" class=" h-40"/>
      </div>

      {/* Forgot password form */}
      <div className="p-8 rounded-lg mt-12">
        <div className="text-center">
          <h2 className="text-2xl font-bold mt-4 pb-4">Forgot Password</h2>
        </div>
        {/* Message for OTP */}
        <p className="mb-14 text-sm text-gray-600 font-bold whitespace-nowrap text-center">
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
            className="w-full p-2 rounded border border-gray-300 mb-4 bg-gray-200"
          />
        </div>
        {/* Use Link from Next.js for navigation */}
        <Link href="/en/otp_verification">
            <button className="w-full bg-green-500 text-white p-2 rounded cursor-pointer hover:bg-green-600">Continue</button>
        </Link>
      </div>
    </div>
  );
};

export default ForgotPasswordPage;
