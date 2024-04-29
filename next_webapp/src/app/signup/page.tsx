// SignUpPage.js
'use client'

import React, { useState } from 'react';
import '../../../public/styles/signup.css';

const SignUpPage = () => {
  const [password, setPassword] = useState('');
  const [passwordStrength, setPasswordStrength] = useState('');

  const handlePasswordChange = (event) => {
      const newPassword = event.target.value;
      setPassword(newPassword);
      setPasswordStrength(checkPasswordStrength(newPassword));
  };

  const checkPasswordStrength = (password) => {
      if (password.length < 6) {
          return 'Weak';
      } else if (password.length < 10) {
          return 'Moderate';
      } else {
          return 'Strong';
      }
  };

  const getPasswordStrengthColor = () => {
      switch (passwordStrength) {
          case 'Weak':
              return 'red';
          case 'Moderate':
              return 'orange';
          case 'Strong':
              return 'green';
          default:
              return 'transparent';
      }
  };
  

  // Calculate the width of the password strength bar based on the length of the password (limited to 10 characters)
  const passwordStrengthStyle = {
    width: `${Math.min((password.length / 10) * 100, 100)}%`,
    backgroundColor: getPasswordStrengthColor(),
};

  return (
      <div className="flex flex-col items-center justify-center min-h-screen">
          {/* Already a member section */}
          <div className="already-member">
              <p className="text-sm mr-2">Already a member?</p>
              <a href="/login" className="login-button">Log In</a>
          </div>

          {/* Logo */}
          <div className="logo">
              {/* Add your logo here */}
              <img src="/img/image.png" alt="Logo" />
          </div>

          {/* Sign-up form */}
          <div className="signup-form">
              <div className="text-center mb-8">
                  <h2 className="text-2xl">Account Sign Up</h2>
              </div>
              <div className="mb-4 flex">
                  <input type="text" placeholder="First name" className="first-name" />
                  <input type="text" placeholder="Last name" className="last-name" />
              </div>
              <div className="mb-4">
                  <input type="email" placeholder="Email" className="email" />
              </div>
              <div className="mb-4">
                  <input type="password" placeholder="Password" className="password" onChange={handlePasswordChange} />
              </div>
              <div className="mb-4">
                  {/* Password strength bar */}
                  <div className="password-strength-bar">
                      <div className="password-strength-bar-inner" style={passwordStrengthStyle}></div>
                  </div>
                  <p className="password-strength-text">Password Strength: {passwordStrength}</p>
              </div>
              <button className="next-button">Next</button>
          </div>
      </div>
  );
};

export default SignUpPage;