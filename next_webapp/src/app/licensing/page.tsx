import Header from "../../components/Header";
import Footer from "../../components/Footer";
import React from 'react';
const Licensing = () => {
  return (
  <div>
  <Header />
  <main>
  <div className="h-[70rem] px-[5rem] content-center font-sans-serif bg-gray-200">
    
  <h1 className="text-black text-4xl left-content w-full md:w-1/2 p-6 md:p-10"><strong>Licensing</strong></h1>
 
  <div className="content-wrapper flex flex-wrap">
    <div className="left-content w-full md:w-1/2 p-6 md:p-10">
      <div>
        <h2 className="text-black text-lg"><strong>Terms of Use</strong></h2>
        <br/>
        <p className="justify-center text-black text-lg">
          By accessing and using our website, you agree to comply with the
          following terms and conditions:
        </p>
        <br />
        <ul className="list-disc pl-5">
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
      </div>
      <br/>
      <div>
        <h2 className=" text-black text-lg"><strong>License Grant</strong></h2>
        <br />
        <p>
          We grant you a limited, non-exclusive, and revocable license to
          access and use our website for personal and non-commercial purposes.
          This license does not include the right to:
        </p>
        <br />
        <ul className="list-disc pl-5">
            <li>Modify, adapt, or reverse engineer any part of the website.</li>
            <li>Copy or distribute content without permission.</li>
            <li>
              Engage in any activity that disrupts the normal functioning of the
              website.
            </li>
          </ul>
          <br/>
        <h2 className="text-black text-lg"><strong>Termination</strong></h2>
        <br />
        <p>
          We reserve the right to terminate your license and access to the
          website at our discretion. Upon termination, you must cease using
          the website, and any provisions of this agreement that should
          survive termination will continue to apply.
        </p>
        <br />
       
      </div>
    </div>

    <div className="right-content justify-self-auto w-full md:w-1/2 p-6 md:p-10">
      <div>
        <h2 className="text-black text-lg"><strong>Intellectual Property</strong></h2>
        <br />
        <p >
          All content on this website, including text, images, and other
          multimedia elements, is owned by MOP. You may not use, reproduce, or
          distribute our content without explicit permission.
        </p>
        <br />
       
        <div className="mt-32 pt-16 mb-5">
        <h2 className="text-black text-lg"><strong>Restrictions</strong></h2>
        <br />
        <p>You may not:</p>
        <br />
        <ul className="list-disc pl-5">
            <li>Use the website for any illegal or unauthorized purpose.</li>
            <li>
              Attempt to gain unauthorized access to our systems or networks.
            </li>
            <li>
              Violate any applicable laws or regulations while using the
              website.
            </li>
          </ul>
          </div>
      </div>
    </div>
  </div>
 
  <div className="contact-info mt-8 text-center">
    <p className="text-black text-lg">
      If you have any questions or concerns about our licensing agreement,
      please contact us at <a href="mailto:licensing@MOP.com.au">licensing@MOP.com.au</a>
    </p>
  </div>
  </div>
  </main>
  <Footer />
</div>

  );
};

export default Licensing;