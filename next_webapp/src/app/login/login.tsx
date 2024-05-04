import React, { useState } from 'react';
import '../../../public/styles/login.css'; // Adjust the path as necessary

function LoginForm() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [passwordVisible, setPasswordVisible] = useState(false);
    const [error, setError] = useState('');

    const handleEmailChange = (event) => {
        setEmail(event.target.value);
    };

    const handlePasswordChange = (event) => {
        setPassword(event.target.value);
    };

    const togglePasswordVisibility = (event) => {
        event.preventDefault(); // Prevents the link from triggering a page load
        setPasswordVisible(!passwordVisible);
    };

    const handleSubmit = (event) => {
        event.preventDefault();
        if (!email || !password) {
            setError('Please fill in both fields');
            return;
        }
        console.log('Submitting', email, password); // Placeholder for authentication logic
        // Reset the error message on successful validation
        setError('');
    };

    const inputLabelStyle = (inputValue) => ({
        transform: inputValue ? 'translateY(-20px)' : 'none',
        fontSize: inputValue ? '12px' : '16px',
        color: inputValue ? '#333' : '#666'
    });

    return (
        <div className="flex flex-col items-center justify-center min-h-screen">
            <form onSubmit={handleSubmit} className="login-form">
                <div className="input-group">
                    <input
                        type="email"
                        id="email"
                        name="email"
                        placeholder="Email"
                        value={email}
                        onChange={handleEmailChange}
                        required
                    />
                    <label htmlFor="email" style={inputLabelStyle(email)}>Email</label>
                </div>
                <div className="input-group relative">
                    <input
                        type={passwordVisible ? 'text' : 'password'}
                        id="password"
                        name="password"
                        placeholder="Password"
                        value={password}
                        onChange={handlePasswordChange}
                        required
                    />
                    <label htmlFor="password" style={inputLabelStyle(password)}>Password</label>
                    <span onClick={togglePasswordVisibility} style={{ cursor: 'pointer', position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)' }}>
                        {passwordVisible ? '👁️‍🗨️' : '👁️'}
                    </span>
                </div>
                <button type="submit" className="login-button">Login</button>
                {error && <div className="error text-red-500">{error}</div>}
            </form>
        </div>
    );
}

export default LoginForm;
