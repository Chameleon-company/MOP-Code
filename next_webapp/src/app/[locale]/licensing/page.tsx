"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import React from "react";
import { useTranslations } from "next-intl";
import { useState } from "react";

import Tooglebutton from "../Tooglebutton/Tooglebutton";

const Licensing = () => {
  const t = useTranslations("licensing");
 //dark theme
 const [dark_value,setdarkvalue] = useState(false);
  
 const handleValueChange = (newValue: boolean | ((prevState: boolean) => boolean))=>{
   setdarkvalue(newValue);
 }
  return (
    <div className= {`${dark_value && "dark"}`}>
      <Header />
      <Tooglebutton onValueChange={handleValueChange}/>
      <main>
        <div className="h-[70rem] px-[5rem] content-center font-sans-serif bg-white dark:bg-pr_bg_dark dark:text-white">
          <h1 className="text-black text-4xl left-content w-full md:w-1/2 p-6 md:p-10 dark:text-white">
            <strong>{t("Licensing")}</strong>
          </h1>

          <div className="content-wrapper flex flex-wrap">
            <div className="left-content w-full md:w-1/2 p-6 md:p-10">
              <div>
                <h2 className="text-black text-lg dark:text-white">
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
                <h2 className="text-black text-lg dark:text-white">
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
                <h2 className="text-black text-lg dark:text-white">
                  <strong>{t("t3")}</strong>
                </h2>
                <br />
                <p>{t("p10")}</p>
                <br />
              </div>
            </div>

            <div className="right-content justify-self-auto w-full md:w-1/2 p-6 md:p-10">
              <div>
                <h2 className="text-black text-lg dark:text-white">
                  <strong>{t("t4")}</strong>
                </h2>
                <br />
                <p>{t("p11")}</p>
                <br />

                <div className="mt-32 pt-16 mb-5">
                  <h2 className="text-black text-lg dark:text-white">
                    <strong>{t("t5")}</strong>
                  </h2>
                  <br />
                  <p>{t("p12")}</p>
                  <br />
                  <ul className="bullet-list">
                    <li>{t("p13")}</li>
                    <li>{t("p14")}</li>
                    <li>{t("p15")}</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="contact-info mt-8 text-center">
            <p className="contact-text">
              {t("p16")}{" "}
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
