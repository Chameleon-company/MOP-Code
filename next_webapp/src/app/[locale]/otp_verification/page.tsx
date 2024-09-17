"use client"

import React, { useState } from 'react';
import '../../../../public/styles/otp_verification.css';
import Header from "../../../components/Header";


const PasswordStrengthIndicator = ({ password }) => {
    const calculateStrength = (password : string) => {
      // You can implement your password strength calculation logic here.
      // For simplicity, I'll just return the length of password
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
    <div className='min-h-screen 'style={{ backgroundColor: 'white' }}>
    <Header/>
    <div className="flex-grow flex flex-col items-center justify-center " >
   
      <div className="text-center">
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
        <div className='mb-6'>
          <input
            type={passwordVisible ? "text" : "password"}
            placeholder="New Password"
            className="reset-input eye-placeholder"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
          />
          <PasswordStrengthIndicator password={newPassword} />
        </div>
        <div className='mb-8'>
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
      
        <button className="reset-button w-full mb-4 w-full max-w-xs sm:max-w-sm md:w-1/2 lg:w-2/3" >Reset Password</button>
      
      </div>
    </div>
    </div>
    
  );
};

export default OTPVerificationPage;
