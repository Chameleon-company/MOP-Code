import React, { useState } from 'react';

function LoginForm() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleEmailChange = (event) => {
        setEmail(event.target.value);
    };

    const handlePasswordChange = (event) => {
        setPassword(event.target.value);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        // add logic to validate or send request to your authentication server
        if (email && password) {
            console.log('Submitting', email, password);
            // Placeholder for authentication logic
        } else {
            setError('Please fill in both fields');
        }
    };

    const emailLabelStyle = {
        transform: email ? 'translateY(-20px)' : 'none',
        fontSize: email ? '12px' : '16px',
        color: email ? '#333' : '#666'
    };

    const passwordLabelStyle = {
        transform: password ? 'translateY(-20px)' : 'none',
        fontSize: password ? '12px' : '16px',
        color: password ? '#333' : '#666'
    };

    return (
        <div className="login-container">
            <form onSubmit={handleSubmit}>
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
                    <label htmlFor="email" style={emailLabelStyle}>Email</label>
                </div>
                <div className="input-group">
                    <input
                        type="password"
                        id="password"
                        name="password"
                        placeholder="Password"
                        value={password}
                        onChange={handlePasswordChange}
                        required
                    />
                    <label htmlFor="password" style={passwordLabelStyle}>Password</label>
                </div>
                <button type="submit">Login</button>
                {error && <div className="error">{error}</div>}
            </form>
        </div>
    );
}

export default LoginForm;
