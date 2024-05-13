// ForgotPasswordPage.js

import React from "react";
import Link from "next/link"; // Import Link from Next.js
import "../../../../public/styles/forgot-password.css";

const ForgotPasswordPage = () => {
  return (
    <div className="first flex flex-col items-center justify-center min-h-screen">
      {/* No Account Yet */}
      <div className="already-member">
        <p className="text-sm mr-2">No Account Yet?</p>
        <a href="/signup" className="login-button text-sm">
          Sign up
        </a>
      </div>

      {/* Logo */}
      <div className="logo">
        {/* Add your logo here */}
        <img src="/img/image.png" alt="Logo" />
      </div>

      {/* Forgot password form */}
      <div className="forgot-password-form">
        <div className="text-center">
          <h2 className="text-2xl">Forgot Password</h2>
        </div>
        {/* Message for OTP */}
        <p className="otp-message">
          <span>ONE-TIME PASSWORD (OTP) WILL BE SENT TO YOUR</span>
          <span>EMAIL ADDRESS FOR VERIFICATION</span>
        </p>
        <div className="mb-0">
          <input type="email" placeholder="Email" className="email" />
        </div>
        {/* Use Link from Next.js for navigation */}
        <a href="/otp_verification">
          <button className="continue-button">continue</button>
        </a>
      </div>
    </div>
  );
};

export default ForgotPasswordPage;
