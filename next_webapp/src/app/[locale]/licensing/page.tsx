import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import React from "react";
import { useTranslations } from "next-intl";

const Licensing = () => {
  const t = useTranslations("licensing");

  return (
    <div>
      <Header />
      <main>
        <div className="h-[70rem] px-[5rem] content-center font-sans-serif bg-gray-200">
          <h1 className="text-black text-4xl left-content w-full md:w-1/2 p-6 md:p-10">
            <strong>{t("Licensing")}</strong>
          </h1>

          <div className="content-wrapper flex flex-wrap">
            <div className="left-content w-full md:w-1/2 p-6 md:p-10">
              <div>
                <h2 className="text-black text-lg">
                  <strong>{t("t1")}</strong>
                </h2>
                <br />
                <p>{t("p1")}</p>
                <br />
                <ul className="bullet-list">
                  <li>{t("p2")}</li>
                  <li>{t("p3")}</li>
                  <li>{t("p4")}</li>
                  <li>{t("p5")}</li>
                </ul>
              </div>
              <br />
              <div>
                <h2 className="text-black text-lg">
                  <strong>{t("t2")}</strong>
                </h2>
                <br />
                <p>{t("p6")}</p>
                <br />
                <ul className="bullet-list">
                  <li>{t("p7")}</li>
                  <li>{t("p8")}</li>
                  <li>{t("p9")}</li>
                </ul>
                <br />
                <h2 className="text-black text-lg">
                  <strong>{t("t3")}Termination</strong>
                </h2>
                <br />
                <p>{t("p10")}</p>
                <br />
              </div>
            </div>

            <div className="right-content justify-self-auto w-full md:w-1/2 p-6 md:p-10">
              <div>
                <h2 className="text-black text-lg">
                  <strong>{t("t4")}</strong>
                </h2>
                <br />
                <p>{t("p11")}</p>
                <br />

                <div className="mt-32 pt-16 mb-5">
                  <h2 className="text-black text-lg">
                    <strong>{t("t5")}</strong>
                  </h2>
                  <br />
                  <p>{t("p1")}You may not:</p>
                  <br />
                  <ul className="bullet-list">
                    <li>
                      {t("p12")}Use the website for any illegal or unauthorized
                      purpose.
                    </li>
                    <li>
                      {t("p13")}Attempt to gain unauthorized access to our
                      systems or networks.
                    </li>
                    <li>
                      {t("p14")}Violate any applicable laws or regulations while
                      using the website.
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="contact-info mt-8 text-center">
            <p className="contact-text">
              {t("p15")}If you have any questions or concerns about our
              licensing agreement, please contact us at{" "}
              <a href="mailto:licensing@MOP.com.au">licensing@MOP.com.au</a>
            </p>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Licensing;
