import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import Image from "next/image";
import React from "react";
import { useTranslations } from "next-intl";

const Licensing = () => {
  const t = useTranslations("licensing");

  return (
    <div className="flex flex-col min-h-screen bg-gray-100">
      <Header />
      <main className="flex-grow mr-[46px]">

              <h1 className="text-black text-4xl w-[287.5px] h-[53.42px] mx-auto my-8 flex items-center justify-center text-center">
                <strong>{t("Licensing")}</strong>
              </h1>

              <div className="ml-4 sm:ml-6 md:ml-[46px]">
                <h2 className="text-black text-xl font-semibold mb-2">
                  {t("t1")}
                </h2>
                <p className="text-black text-base mb-4">{t("p1")}</p>
                <ul className="list-disc pl-6 text-black text-base mb-8">
                  <li className="mb-2">{t("p2")}</li>
                  <li className="mb-2">{t("p3")}</li>
                  <li className="mb-2">{t("p4")}</li>
                  <li className="mb-2">{t("p5")}</li>
                </ul>
              </div>

              <div className="ml-4 sm:ml-6 md:ml-[46px]">
                <h2 className="text-black text-xl font-semibold mb-2">
                  {t("t2")}
                </h2>
                <p className="text-black text-base mb-4">{t("p6")}</p>
                <ul className="list-disc pl-6 text-black text-base mb-8">
                  <li className="mb-2">{t("p7")}</li>
                  <li className="mb-2">{t("p8")}</li>
                  <li className="mb-2">{t("p9")}</li>
                </ul>
              </div> 

              <div className="ml-4 sm:ml-6 md:ml-[46px]">
                <h2 className="text-black text-xl font-semibold mb-2">
                  {t("t4")}
                </h2>
                <p className="text-black text-base mb-6">{t("p11")}</p>
              </div>

              <div className="ml-4 sm:ml-6 md:ml-[46px]">
                  <h2 className="text-black text-xl font-semibold mb-2">
                    {t("t5")}
                  </h2>
                  <p className="text-black text-base mb-2">{t("p12")}</p>
                  <ul className="list-disc pl-6 text-black text-base">
                    <li className="mb-2">{t("p13")}</li>
                    <li className="mb-2">{t("p14")}</li>
                    <li className="mb-6">{t("p15")}</li>
                  </ul>
              </div>

            <div className="ml-4 sm:ml-6 md:ml-[46px]">
                <h2 className="text-black text-xl font-semibold mb-2">
                  {t("t3")}
                </h2>
                <p className="text-black text-base mb-8">{t("p10")}</p>
            </div>

          <div className="contact-info mt-8 text-center">
            <p className="text-black text-base mb-2">
              {t("p16")}{" "}
              <a href="mailto:licensing@MOP.com.au" className="text-blue-600">
                licensing@MOP.com.au
              </a>
            </p>
          </div>
          <a href="/url">
            <Image
              src="/img/chatbot.png"
              alt="Chatbot Icon"
              width={61.87}
              height={61.75}
              className="ml-auto mr-24 mb-3"
            />
          </a>
      </main>
      <Footer />
    </div>
  );
};

export default Licensing;

