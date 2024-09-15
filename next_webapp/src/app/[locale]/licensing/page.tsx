import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import React from "react";
import { useTranslations } from "next-intl";

const Licensing = () => {
  const t = useTranslations("licensing");

  return (
    <div className="flex flex-col min-h-screen bg-gray-100 bg-white dark:bg-zinc-800 dark:text-slate-100">
      <Header />
      <main className="flex-grow">
        <div className="h-[70rem] px-[5rem] content-center font-sans-serif">
          <h1 className=" text-4xl left-content w-full md:w-1/2 p-6 md:p-10">
            <strong>{t("Licensing")}</strong>
          </h1>

          <div className="content-wrapper flex flex-wrap">
            <div className="left-content w-full md:w-1/2 p-6 md:p-10">
              <div>
                <h2 className=" text-xl font-semibold mb-2">
                  {t("t1")}
                </h2>
                <p className=" text-base mb-4">{t("p1")}</p>
                <ul className="list-disc pl-6  text-base mb-8">
                  <li className="mb-2">{t("p2")}</li>
                  <li className="mb-2">{t("p3")}</li>
                  <li className="mb-2">{t("p4")}</li>
                  <li className="mb-2">{t("p5")}</li>
                </ul>
              </div>
              <div>
                <h2 className=" text-xl font-semibold mb-2">
                  {t("t2")}
                </h2>
                <p className=" text-base mb-4">{t("p6")}</p>
                <ul className="list-disc pl-6  text-base mb-8">
                  <li className="mb-2">{t("p7")}</li>
                  <li className="mb-2">{t("p8")}</li>
                  <li className="mb-2">{t("p9")}</li>
                </ul>
                <h2 className=" text-xl font-semibold mb-2">
                  {t("t3")}
                </h2>
                <p className=" text-base mb-8">{t("p10")}</p>
              </div>
            </div>

            <div className="right-content justify-self-auto w-full md:w-1/2 p-6 md:p-10">
              <div>
                <h2 className=" text-xl font-semibold mb-2">
                  {t("t4")}
                </h2>
                <p className=" text-base mb-8">{t("p11")}</p>

                <div className="mt-32 pt-16 mb-5">
                  <h2 className=" text-xl font-semibold mb-2">
                    {t("t5")}
                  </h2>
                  <p className=" text-base mb-4">{t("p12")}</p>
                  <ul className="list-disc pl-6  text-base">
                    <li className="mb-2">{t("p13")}</li>
                    <li className="mb-2">{t("p14")}</li>
                    <li className="mb-2">{t("p15")}</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="contact-info mt-8 text-center">
            <p className=" text-base">
              {t("p16")}{" "}
              <a href="mailto:licensing@MOP.com.au" className="text-blue-600">
                licensing@MOP.com.au
              </a>
            </p>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Licensing;
