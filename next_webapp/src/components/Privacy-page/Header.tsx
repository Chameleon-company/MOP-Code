import React from 'react';
import './Header.css';

const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="logo">Logo</div>
      <nav>
        <ul>
          <li>Home</li>
          <li>About Us</li>
          <li>Usecases</li>
          <li>Statistics</li>
          <li>Upload</li>
        </ul>
      </nav>
      <div className="buttons">
        <button>Language</button>
        <button>Sign Up</button>
        <button>Log In</button>
      </div>
    </header>
  );
};

export default Header;
