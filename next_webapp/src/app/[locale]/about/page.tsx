"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/about.css";
import { useTranslations } from "next-intl";
import { useState } from "react";
import Tooglebutton from "../Tooglebutton/Tooglebutton";

const About = () => {
  const t = useTranslations("about");

  //dark theme
  const [dark_value, setdarkvalue] = useState(false);
  const handleValueChange = (
    newValue: boolean | ((prevState: boolean) => boolean)
  ) => {
    setdarkvalue(newValue);
  };
  return (
    <div className={`${dark_value && "dark"}`}>
      <div className="bg-white dark:bg-black">
        <Header />
        <div className="container bg-green-600">
          <div className="content">
            <img src="/img/mel.jpg" alt="Melbourne Open Playground" className="float-right border-neutral-600"/>
            <div className="text-box">
              <h2 className="">{t("About Us")}</h2>
              <br />
              <p className="w-full md:w-[40%]">{t("p2")}</p>
            </div>
          </div>
        </div>

        <div className="container">
          <div className="content">
            <img src="/img/leadership.png" alt="Leadership Image" className="float-right border-neutral-600 dark:border-white"/>
            <div className="text-box">
              <h2 className="text-black dark:text-white">{t("Open Data Leadership")}</h2>
              <br />
              <p className="text-black dark:text-white w-full md:w-[40%]">{t("p3")}</p>
            </div>
          </div>
        </div>

        <div className="container bg-green-600">
          <div className="content">
            <img src="/img/goals.png" alt="Our Goals" className="float-left mr-4 border-neutral-600"/>
            <div className="text-box">
              <h2 className="">{t("Our Goals")}</h2>
              <br />
              <p className="">{t("p4")}</p>
            </div>
          </div>
        </div>

        <Tooglebutton onValueChange={handleValueChange} />
        <Footer />
      </div>
    </div>
  );
};

export default About;
