import React, { useState } from 'react';
import { Helmet } from 'react-helmet';
import '../../../public/styles/login.css'; // Ensure the path is correct

function LoginPage() {
    const [formData, setFormData] = useState({
        email: '',
        password: ''
    });
    const [passwordVisible, setPasswordVisible] = useState(false);
    const [error, setError] = useState('');

    const handleInputChange = (event) => {
        const { name, value } = event.target;
        setFormData(prevState => ({
            ...prevState,
            [name]: value
        }));
    };

    const togglePasswordVisibility = () => {
        setPasswordVisible(prevState => !prevState);
    };

    const handleSubmit = (event) => {
        event.preventDefault();
        const { email, password } = formData;
        if (!email || !password) {
            setError('Please fill in both fields');
            return;
        }
        console.log('Submitting', email, password); 
        setError(''); 
    };

    const getLabelStyle = (inputValue) => ({
        transform: inputValue ? 'translateY(-20px)' : 'none',
        fontSize: inputValue ? '12px' : '16px',
        color: inputValue ? '#333' : '#666'
    });

    return (
        <>
            <Helmet>
                <meta charSet="UTF-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                <title>Login Page</title>
                <link rel="stylesheet" href="login.css" />
            </Helmet>
            <div className="top-bar">
                <img src="logo.png" alt="Chameleon Logo" className="logo"/>
                <div className="signup-container">
                    <span className="no-account">No Account yet?</span>
                    <button className="sign-up-button">Sign Up</button>
                </div>
            </div>
            <div className="login-container">
                <h2>Account Log In</h2>
                <p className="subtitle">Please login to continue to your account</p>
                <form onSubmit={handleSubmit} className="login-form">
                    <div className="input-group">
                        <input
                            type="email"
                            id="email"
                            name="email"
                            placeholder="Email"
                            value={formData.email}
                            onChange={handleInputChange}
                            required
                        />
                        <label htmlFor="email" style={getLabelStyle(formData.email)}>Email</label>
                    </div>
                    <div className="input-group" style={{ position: 'relative' }}>
                        <input
                            type={passwordVisible ? 'text' : 'password'}
                            id="password"
                            name="password"
                            placeholder="Password"
                            value={formData.password}
                            onChange={handleInputChange}
                            required
                        />
                        <label htmlFor="password" style={getLabelStyle(formData.password)}>Password</label>
                        <span onClick={togglePasswordVisibility} style={{ cursor: 'pointer', position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)' }}>
                            {passwordVisible ? '👁️‍🗨️' : '👁️'}
                        </span>
                    </div>
                    <button type="submit" className="login-button">Login</button>
                    {error && <div className="error">{error}</div>}
                </form>
            </div>
        </>
    );
}

export default LoginPage;
