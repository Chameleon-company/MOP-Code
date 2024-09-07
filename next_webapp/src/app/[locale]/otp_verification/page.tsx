"use client"

import React, { useState } from 'react';
import '../../../../public/styles/otp_verification.css';

const PasswordStrengthIndicator = ({ password }) => {
    const calculateStrength = (password) => {
      // You can implement your password strength calculation logic here
      // For simplicity, I'll just return the length of the password
      return password.length;
    };
  
    const strength = password ? calculateStrength(password) : 0;
  
    let strengthText = password ? (strength < 6 ? "Weak" : strength < 10 ? "Moderate" : "Strong") : "Password-Strength";
    let strengthColor;
  
    if (strength < 6) {
      strengthColor = 'red';
    } else if (strength < 10) {
      strengthColor = 'orange';
    } else {
      strengthColor = 'green';
    }
  
    return (
      <div className="password-strength">
        <div
          className="password-strength-bar"
          style={{ width: `${(strength / 20) * 100}%`, backgroundColor: strengthColor }}
        ></div>
        <div className="password-strength-text">{strengthText}</div>
      </div>
    );
  };
  
  
  
  

const OTPVerificationPage = () => {
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const togglePasswordVisibility = () => {
    setPasswordVisible(!passwordVisible);
  };

  return (
    <div className="first flex flex-col items-center justify-center min-h-screen" style={{ backgroundColor: 'white' }}>
      {/* No Account Yet */}
      <div className="already-member">
        <p className="text-sm mr-2">No Account Yet?</p>
        <a href="/signup" className="signup-button text-sm">Sign up</a>
      </div>

      {/* Logo */}
      <div className="logo">
        {/* Add your logo here */}
        <img src="/img/new-logo-green.png" alt="Chameleon Logo" className=" h-40" />
      </div>

      <div className="text-center mt-4">
        <h2 className="text-2xl font-bold">OTP VERIFICATION</h2>
      </div>
      {/* Message for OTP */}
      <p className="otp-message">
        <span>We have sent a four-digit code</span>
        <span>to your email address</span>
      </p>
      <div className="password-reset-form ">
        <div className='mb-4'>
          <input type="text" placeholder="Four Digit Code" className="reset-input" />
        </div>
        <div className='mb-4'>
          <input
            type={passwordVisible ? "text" : "password"}
            placeholder="New Password"
            className="reset-input eye-placeholder"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
          />
          <PasswordStrengthIndicator password={newPassword} />
        </div>
        <div className='mb-4'>
          <input
            type={passwordVisible ? "text" : "password"}
            placeholder="Confirm Password"
            className="reset-input eye-placeholder"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
          />
          {newPassword && confirmPassword && newPassword !== confirmPassword && (
            <div style={{ color: 'red' }}>Passwords do not match</div>
          )}
          <PasswordStrengthIndicator password={newPassword} />
        </div>
      </div>

      <button className="reset-button" >Reset Password</button>

    </div>
  );
};

export default OTPVerificationPage;
