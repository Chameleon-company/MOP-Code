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
  const [dark_value,setdarkvalue] = useState(false);
  
  const handleValueChange = (newValue: boolean | ((prevState: boolean) => boolean))=>{
    setdarkvalue(newValue);
  }
  return (
    <div className= {`${dark_value && "dark"}`}>
    <div className="bg-white dark:bg-black">
      <Header />
      <Tooglebutton onValueChange={handleValueChange}/>
      <div className="text-black dark:text-white text-[120px] mb-1 mt-1 sm:ml-72">{t("About")}</div>
      <div className="text-black dark:text-white text-[120px] mb-1 -mt-6 sm:ml-72">{t("Us")}</div>
      <div className="float-right absolute pl-10 top-0 right-0 mt-40 image-2 overflow-hidden image-container before:dark:bg-white after:dark:bg-white">
        <img src="/img/mel.jpg" alt="About Us Image" className="w-full ml-2 mt-36 h-auto"/>
      </div>

      <div className="w-full bg-gray-100 border-y-2  py-2 text-center mt-96 text-black dark:bg-[#131619] dark:text-white">
        <h2 className="text-4xl font-semibold p-5">{t("About MOP")}</h2>
        <p className="font-normal text-xl mb-3 p-5">{t("p1")}</p>
      </div>

      <div className="m-4 bg-white dark:bg-black">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mx-10 mt-10">
          <div
             style={{ backgroundColor: "rgb(255,255,255)", color: "black" }}
            className="flex flex-col items-center p-4 border-2 shadow-xl rounded-sm h-full"
          >
            <div className="flex flex-row h-full">
              <span className="font-bold m-2 text-3xl">
                {t("About")} <br />
                {t("Us")}
              </span>
              <div className="mt-10 pl-8 w-56 h-44 relative">
                <img
                  src="/img/about-us.png"
                  className="absolute inset-0 w-full h-full object-cover"
                  alt="Description of the image"
                />
                <div className="absolute -top-3 right-0 h-3 w-1/2 bg-black"></div>
                <div className="absolute bottom-0 -left-3 h-1/2 bg-black w-3"></div>
              </div>
            </div>

            <p className="p-4 text-center">
              <span className="text-wrap">{t("p2")}</span>
            </p>
          </div>
          <div
            style={{ backgroundColor: "rgb(74 222 128)", color: "black" }}
            className="flex flex-col items-center border-black shadow-xl  p-4 h-full"
          >
            <div className="flex flex-row h-full">
              <span className="font-bold m-2 text-3xl">
                {t("Open Data Leadership")}
              </span>
              <div className="mt-10 pl-8 w-56 h-44 relative flex-shrink-0">
                <img
                  src="/img/leadership.png"
                  className="absolute inset-0 w-full h-full object-cover"
                  alt="Description of the image"
                />
                <div className="absolute -top-3 right-0 h-3 w-1/2 bg-black"></div>
                <div className="absolute bottom-0 -left-3 h-1/2 bg-black w-3"></div>
              </div>
            </div>

            <p className="p-4 text-center">
              <span className="text-wrap">{t("p3")}</span>
            </p>
          </div>
          <div
            style={{ backgroundColor: "rgb(255,255,255)", color: "black" }}
            className="flex flex-col items-center p-4 border-2 shadow-xl rounded-sm h-full"
          >
            <div className="flex flex-row h-64">
              <span className="font-bold m-2 text-3xl">
                {t("Our")} <br />
                {t("Goals")}
              </span>
              <div className="mt-10 pl-8 w-56 h-44 relative">
                <img
                  src="/img/goals.png"
                  className="absolute inset-0 w-full h-full object-cover"
                  alt="Description of the image"
                />
                <div className="absolute -top-3 right-0 h-3 w-1/2 bg-black"></div>
                <div className="absolute bottom-0 -left-3 h-1/2 bg-black w-3"></div>
              </div>
            </div>

            <p className="p-4 text-center">
              <span className="text-wrap">{t("p4")}</span>
            </p>
          </div>
        </div>
      </div>

      <Footer />
    </div>
    </div>
  );
};

export default About;
