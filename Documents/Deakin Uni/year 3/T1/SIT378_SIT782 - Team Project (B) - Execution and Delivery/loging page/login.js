import React, { useState } from 'react';

function LoginForm() {
    const [email, setEmail] = useState('');

    const handleEmailChange = (event) => {
        setEmail(event.target.value);
    };

    const emailLabelStyle = {
        transform: email ? 'translateY(-20px)' : 'none',
        fontSize: email ? '12px' : '16px',
        color: email ? '#333' : '#666'
    };

    return (
        <div className="login-container">
            <form>
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
            </form>
        </div>
    );
}

export default LoginForm;
