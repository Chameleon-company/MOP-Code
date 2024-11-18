import React from "react";
import { Link } from "react-router-dom";
import dashboardIcon from '../assets/header-logo.png';
import dashboardBackground from '../assets/dashboard-background.png';
import aboutIcon from '../assets/about-icon.png';
import caseIcon from '../assets/case-icon.png';
import resourceIcon from '../assets/resource-icon.png';
import dataIcon from '../assets/data-icon.png';
import contactIcon from '../assets/contact-icon.png';

const dashboardStyles = {
    dashboardContainer: {
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        fontFamily: "Poppins",
    },
    dashboardIntro: {
        width: "100%",
        height: "500px",
        backgroundImage: `url(${dashboardBackground})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        textAlign: 'center',
        color: 'white',
    },
    iconStyle: {
        height: '100px',
        position: 'absolute',
        top: '-50px',
        left: '50%',
        transform: 'translateX(-50%)',
    },

    textBoxWrapper: {
        position: 'relative',
        display: 'flex',
        justifyContent: 'center',
    },
    textBoxStyle: {
        display: 'flex',
        flexDirection: 'column',
        width: '400px',
        height: '300px',
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(0, 0, 0, 0.3)',
        color: '#FFFFFF',
        border: '1px solid rgba(0, 0, 0, 0.6)',
        borderRadius: '10px',
        boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
        padding: '20px',
        boxSizing: 'border-box',
    },
    textLine1Style: {
        fontWeight: '900',
        fontSize: '32px',
    },
    textLine2Style: {
        fontWeight: '900',
        fontSize: '38px',
    },
    textLine3Style: {
        fontSize: '32px',
    },
    textLine4Style: {
        fontWeight: '100',
        fontSize: '20px',
    },
    dashboardMain: {
        width: '100%',
        display: 'flex',
        justifyContent: "space-between",
        alignItems: 'center',
        height: '200px',
        backgroundColor: '#E2F0DC',
        padding: '0 40px'
    },
    navItem: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textDecoration: 'none',
    },
    navIcon: {
        height: '120px',
        marginBottom: '8px',
    },
    navText: {
        color: '#000000',
        fontSize: '24px',
        fontWeight: 'bold',
    },

};

const NavItem = ({ to, icon, label }) => (
    <Link to={to} style={dashboardStyles.navItem}>
        <img src={icon} alt={label} style={dashboardStyles.navIcon} />
        <span style={dashboardStyles.navText}>{label}</span>
    </Link>
);
const Dashboard = () => {
    const navItems = [
        { to: "/about", icon: aboutIcon, label: "About Us" },
        { to: "/casestudies", icon: caseIcon, label: "Case Studies" },
        { to: "/resource-center", icon: resourceIcon, label: "Resource Center" },
        { to: "/datasets", icon: dataIcon, label: "Data Collection" },
        { to: "/contact", icon: contactIcon, label: "Contact Us" },
    ];

    return (
        <div style={dashboardStyles.dashboardContainer}>
            <div style={dashboardStyles.dashboardIntro}>
                <div style={dashboardStyles.textBoxWrapper}>
                    <img src={dashboardIcon} alt="Icon" style={dashboardStyles.iconStyle} />
                    <div style={dashboardStyles.textBoxStyle}>
                        <p style={dashboardStyles.textLine1Style}>MELBOURNE</p>
                        <p style={dashboardStyles.textLine2Style}>OPEN DATA</p>
                        <p style={dashboardStyles.textLine3Style}> PROJECT</p>
                        <p style={dashboardStyles.textLine4Style}>Unlocking Melbourne's Potential through Open Data Innovation.</p>
                    </div>
                </div>
            </div>
            <div style={dashboardStyles.dashboardMain}>
                {navItems.map(item => (
                    <NavItem key={item.label} to={item.to} icon={item.icon} label={item.label} />
                ))}
            </div>
        </div>
    );
};

export default Dashboard;
