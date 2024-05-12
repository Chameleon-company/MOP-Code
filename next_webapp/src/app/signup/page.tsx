// SignUpPage.js
'use client'

import React, { useState } from 'react';

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
            <div className="already-member">
                {/* Already a member section */}
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
                    <input type="text" placeholder="First name" className="first-name" id="firstName" />
                    <input type="text" placeholder="Last name" className="last-name" id="lastName" />
                </div>
                <div className="mb-4">
                    <input type="email" placeholder="Email" className="email" id="email" />
                </div>
                <div className="mb-4">
                    <input type="password" placeholder="Password" className="password" onChange={handlePasswordChange} id="password" />
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

  return (
      <div className="flex flex-col items-center justify-center min-h-screen">
          {/* Already a member section */}
          <div className="absolute top-20 right-20 ml-4 m-4 flex items-center text-[#666666]">
              <p className="text-sm mr-2">Already a member?</p>
              <a href="/login" className="border border-gray-600 text-gray-600 px-4 py-2 ml-4 hover:border-[#999999]">Log In</a>
          </div>

          {/* Logo */}
          <div className="absolute top-20 left-20 mr-4 m-4">
              {/* Add your logo here */}
              <img src="/img/image.png" alt="Logo" />
          </div>

          {/* Sign-up form */}
          <div className="p-8 rounded-lg mt-12">
              <div className="text-center mb-8">
                  <h2 className="text-2xl font-bold mb-12">Account Sign Up</h2>
              </div>
              <div className="mb-4 flex">
                  <input type="text" placeholder="First name" className="w-1/2 p-2 rounded-md border-solid border-2 border-[#ccc] mr-2 bg-[#e9ebeb]" />
                  <input type="text" placeholder="Last name" className="w-1/2 p-2 rounded-md border-solid border-2 border-[#ccc] mr-2 bg-[#e9ebeb]" />
              </div>
              <div className="mb-4">
                  <input type="email" placeholder="Email" className="w-full p-2 rounded-md border-solid border-2 border-[#ccc] mr-2 bg-[#e9ebeb]" />
              </div>
              <div className="mb-4">
                  <input type="password" placeholder="Password" className="w-full p-2 rounded-md border-solid border-2 border-[#ccc] mr-2 bg-[#e9ebeb]" onChange={handlePasswordChange} />
              </div>
              <div className="mb-4">
                  {/* Password strength bar */}
                  <div className="w-full h-2 border border-gray-300 rounded mb-2">
                      <div className="h-full rounded" style={passwordStrengthStyle}></div>
                  </div>
                  <p className="text-right text-sm text-gray-600 mt-2">Password Strength: {passwordStrength}</p>
              </div>
              <button className="w-full bg-green-500 text-white py-2 px-2 rounded-md cursor-pointer">Next</button>
          </div>
      </div>
  );
};

export default SignUpPage;
