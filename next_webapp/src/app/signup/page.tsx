// SignUpPage.js

import React from 'react';
import '../../../public/styles/signup.css';

const SignUpPage = () => {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen">
        {/* Already a member section */}
        <div className="already-member">
          <p className="text-sm mr-2">Already a member?</p>
          <button className="login-button">Log In</button>
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
            <input type="password" placeholder="Password" className="password" />
          </div>
          <div className="mb-4">
            {/* Password strength bar (you can implement this with JS) */}
            <div className="password-strength"></div>
            <p className="password-strength-text">Password Strength</p>
          </div>
          <button className="next-button">Next</button>
        </div>
      </div>
    );
  };
  
  export default SignUpPage;
















