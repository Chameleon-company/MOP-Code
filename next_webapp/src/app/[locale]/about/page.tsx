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
      <div className="flex flex-row h-full w-full">
        <span className="about-us dark:text-white text-2xl sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl mt-7 ml-4 sm:ml-12 md:ml-16">
        {t("About")} <br />
        {t("Us")}
        </span>
        <div className="mt-7 pl-10 float-right relative flex top-0 right-0 ms-auto overflow-hidden before:dark:bg-white after:dark:bg-white">
          <img src="/img/mel.jpg" alt="About Us Image" className="w-full h-full object-scale-down"/>
          <div className="absolute top-[-0] right-0 h-3 w-1/2 bg-black"></div>
          <div className="absolute bottom-0 left-[-3] h-1/2 bg-black w-3"></div>
        </div>
      </div>

      <div className="w-full bg-gray-100 border-y-2 py-2 text-center mt-12 text-black dark:bg-[#131619] dark:text-white">
        <h2 className="text-lg sm:text-xl md:text-2xl lg:text-3xl xl:text-4xl font-semibold p-3 md:p-5">{t("About MOP")}</h2>
        <p className="text-xs sm:text-sm md:text-md lg:text-lg xl:text-xl m-5 px-3">{t("p1")}</p>
      </div>

      <div className="m-4 bg-white dark:bg-black">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mx-10 mt-10">
          <div
             style={{ backgroundColor: "rgb(255,255,255)", color: "black" }}
            className="flex flex-col items-center p-4 border-2 shadow-xl rounded-sm h-full"
          >
            <div className="flex flex-row h-auto w-full">
              <span className="font-bold m-2 text-lg sm:text-xl md:text-2xl lg:text-3xl xl:text-4xl">
                {t("About")} <br />
                {t("Us")}
              </span>
              <div className="mt-10 float-right relative flex top-0 right-0 w-40 h-36 ms-auto">
                <img
                  src="/img/about-us.png"
                  className="w-full h-full object-cover"
                  alt="Description of the image"
                />
                <div className="absolute -top-3 right-0 h-3 w-1/2 bg-black"></div>
                <div className="absolute bottom-0 -left-3 h-1/2 bg-black w-3"></div>
              </div>
            </div>

            <p className="p-4 text-center ">
              <span className="text-wrap text-xs sm:text-sm md:text-md lg:text-lg xl:text-xl">{t("p2")}</span>
            </p>
          </div>
          <div
            style={{ backgroundColor: "rgb(74 222 128)", color: "black" }}
            className="flex flex-col items-center border-black border-2 rounded-sm shadow-xl p-4 h-full"
          >
            <div className="flex flex-row h-auto w-full">
              <span className="font-bold m-2 text-lg sm:text-xl md:text-2xl lg:text-3xl xl:text-4xl">
                {t("Open Data Leadership")}
              </span>
              <div className="mt-10 float-right relative flex top-0 right-0 w-40 h-36 ms-auto">
                <img
                  src="/img/leadership.png"
                  className="w-full h-full object-cover"
                  alt="Description of the image"
                />
                <div className="absolute -top-3 right-0 h-3 w-1/2 bg-black"></div>
                <div className="absolute bottom-0 -left-3 h-1/2 bg-black w-3"></div>
              </div>
            </div>

            <p className="p-4 text-center">
              <span className="text-wrap text-xs sm:text-sm md:text-md lg:text-lg xl:text-xl">{t("p3")}</span>
            </p>
          </div>
          <div
            style={{ backgroundColor: "rgb(255,255,255)", color: "black" }}
            className="flex flex-col items-center p-4 border-2 shadow-xl rounded-sm h-full"
          >
            <div className="flex flex-row h-auto w-full">
              <span className="font-bold m-2 text-lg sm:text-xl md:text-2xl lg:text-3xl xl:text-4xl">
                {t("Our")} <br />
                {t("Goals")}
              </span>
              <div className="mt-10 float-right relative flex top-0 right-0 w-40 h-36 ms-auto">
                <img
                  src="/img/goals.png"
                  className="w-full h-full object-cover"
                  alt="Description of the image"
                />
                <div className="absolute -top-3 right-0 h-3 w-1/2 bg-black"></div>
                <div className="absolute bottom-0 -left-3 h-1/2 bg-black w-3"></div>
              </div>
            </div>

            <p className="p-4 text-center">
              <span className="text-wrap text-xs sm:text-sm md:text-md lg:text-lg xl:text-xl">{t("p4")}</span>
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