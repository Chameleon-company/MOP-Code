import React from "react";
import Header from "../../components/Header";
import Footer from "../../components/Footer";
import "./licensing.css";

const Licensing = () => {
  return (
    <div className="licensing-page">
      <Header />

      <h1>Licensing Agreement</h1>
      <div className=" pading">
        <div className="content">
          <p>
            Welcome to the licensing page of MOP. This page outlines the terms
            and conditions under which you are granted a license to use our
            website and its content.
          </p>

          <h2>Terms of Use</h2>
          <p>
            By accessing and using our website, you agree to comply with the
            following terms and conditions:
          </p>

          <ul>
            <li>
              You must use the website in accordance with applicable laws and
              regulations.
            </li>
            <li>
              Unauthorized use, reproduction, or distribution of our content is
              strictly prohibited.
            </li>
            <li>
              You are responsible for maintaining the confidentiality of your
              login information.
            </li>
            <li>
              We reserve the right to suspend or terminate your access to the
              website if you violate these terms.
            </li>
          </ul>

          <h2>Intellectual Property</h2>
          <p>
            All content on this website, including text, images, and other
            multimedia elements, is owned by MOP. You may not use, reproduce, or
            distribute our content without explicit permission.
          </p>

          <h2>License Grant</h2>
          <p>
            We grant you a limited, non-exclusive, and revocable license to
            access and use our website for personal and non-commercial purposes.
            This license does not include the right to:
          </p>

          <ul>
            <li>Modify, adapt, or reverse engineer any part of the website.</li>
            <li>Copy or distribute content without permission.</li>
            <li>
              Engage in any activity that disrupts the normal functioning of the
              website.
            </li>
          </ul>

          <h2>Restrictions</h2>
          <p>You may not:</p>
          <ul>
            <li>Use the website for any illegal or unauthorized purpose.</li>
            <li>
              Attempt to gain unauthorized access to our systems or networks.
            </li>
            <li>
              Violate any applicable laws or regulations while using the
              website.
            </li>
          </ul>

          <h2>Termination</h2>
          <p>
            We reserve the right to terminate your license and access to the
            website at our discretion. Upon termination, you must cease using
            the website, and any provisions of this agreement that should
            survive termination will continue to apply.
          </p>
        </div>

        <div className="contact-info">
          <p>
            If you have any questions or concerns about our licensing agreement,
            please contact us at licensing@MOP.com.au
          </p>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default Licensing;
