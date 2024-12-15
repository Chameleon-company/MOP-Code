import React from "react";
import { Link } from "react-router-dom";

const headerStyles = {
    headerContainer: {
        fontFamily: "Poppins",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        backgroundColor: "#E2F0DC", 
        padding: "10px 90px",
    },
    logoContainer: {
        display: "flex",
        alignItems: "center",
    },
    logo: {
        height: "70px",
    },
    logoTextContainer: {
        marginLeft:"20px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
    },
    logoTextBig: {
        fontSize: "24px",
        fontWeight: "bold",
    },
    logoTextSmall: {
        fontSize: "14px",
    },
    navContainer: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
    },
    navLink: {
        textDecoration: "none",
        color: "#000000",
        fontSize: "20px",
        padding: "0 15px",
        margin: "0 10px",
    },
};

const HeaderBackup = () => {
    return (
        <header style={headerStyles.headerContainer}>
            <div style={headerStyles.logoContainer}>
                <img
                    style={headerStyles.logo}
                    src="src/assets/header-logo.png"
                    alt="MOP Logo"
                />
                <div style={headerStyles.logoTextContainer}>
                    <span style={headerStyles.logoTextBig}>MELBOURNE</span>
                    <span style={headerStyles.logoTextSmall}>OPEN DATA PROJECT</span>
                </div>
            </div>
            <div style={headerStyles.navContainer}>
                <Link to="/" style={headerStyles.navLink}>Home</Link>
                <Link to="/casestudies" style={headerStyles.navLink}>Case Studies</Link>
                <Link to="/datasets" style={headerStyles.navLink}>Datasets</Link>
            </div>
        </header>
    );
};

export default HeaderBackup;
