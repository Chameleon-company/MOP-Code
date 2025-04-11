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
      <div className="bg-white dark:bg-[#1d1919]">
        <Header />
        <div className="h-full content-center">
          <div className="container-title bg-green-500 dark:bg-green-900">
            <div className="content-title md:py-8">
              <img
                src="/img/mel.jpg"
                alt="Melbourne Open Playground"
                className="float-left img-section pr-[5%]"
              />
              <div className="text-box font-sans">
                <h2 className="font-bold">{t("About Us")}</h2>
                <br />
                <p className="w-full md:w-[60%]" id="p2">
                  {t("p2")}
                </p>
              </div>
            </div>
          </div>

          <div className="container bg-white dark:bg-[#1d1919]">
            <div className="content-section md:pt-16">
              <img
                src="/img/leadership.png"
                alt="Leadership Image"
                className="float-left img-section pr-[5%]"
              />
              <div className="text-box font-sans">
                <h2 className="font-bold text-black dark:text-white">
                  {t("Open Data Leadership")}
                </h2>
                <br />
                <p
                  className="text-black dark:text-white w-full md:w-[60%]"
                  id="p3"
                >
                  {t("p3")}
                </p>
              </div>
            </div>
          </div>

          <div className="container bg-white  dark:bg-[#1d1919]">
            <div className="content-section md:pt-6">
              <img
                src="/img/goals.png"
                alt="Our Goals"
                className="float-left img-section pr-[5%]"
              />
              <div className="text-box font-sans">
                <h2 className="font-bold text-black dark:text-white">
                  {t("Our Goals")}
                </h2>
                <br />
                <p className="text-black dark:text-white" id="p4">
                  {t("p4")}
                </p>
              </div>
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


// "use client";
// import Header from "../../../components/Header";
// import Footer from "../../../components/Footer";
// import "../../../../public/styles/about.css";
// import { useTranslations } from "next-intl";
// import { useState } from "react";
// import Tooglebutton from "../Tooglebutton/Tooglebutton";

// const About = () => {
//   const t = useTranslations("about");

//   const [dark_value, setdarkvalue] = useState(false);
//   const handleValueChange = (newValue: boolean | ((prevState: boolean) => boolean)) => {
//     setdarkvalue(newValue);
//   };

//   return (
//     <div className={`${dark_value && "dark"}`}>
//       <div className="bg-white dark:bg-[#1d1919] min-h-screen">
//         <Header />

//         {/* ABOUT SECTION */}
//         <section className="px-6 py-12 md:flex md:items-center md:justify-between bg-white dark:bg-[#1d1919]">
//           <div className="md:w-1/2">
//             <h2 className="text-3xl font-bold text-black dark:text-white">{t("About Us")}</h2>
//             <p className="mt-4 text-black dark:text-white">{t("p1")}</p>
//             <p className="mt-2 text-black dark:text-white">{t("p2")}</p>
//           </div>
//           <div className="md:w-1/2 mt-8 md:mt-0 flex justify-center">
//             <img src="/img/mel.jpg" alt="Melbourne" className="rounded-lg w-3/4" />
//           </div>
//         </section>

//         {/* OPEN DATA LEADERSHIP */}
//         <section className="px-6 py-12 text-center bg-gray-100 dark:bg-[#2b2626]">
//           <h2 className="text-3xl font-bold text-black dark:text-white">{t("Open Data Leadership")}</h2>
//           <div className="mt-6 flex flex-col items-center">
//             <img src="/img/leadership.png" alt="Leadership" className="rounded-lg w-3/4" />
//             <p className="mt-4 w-full md:w-[70%] text-black dark:text-white">{t("p3")}</p>
//           </div>
//         </section>

//         {/* GOALS SECTION */}
//         <section className="px-6 py-12 md:flex md:items-center md:justify-between bg-white dark:bg-[#1d1919]">
//           <div className="md:w-1/2 flex justify-center">
//             <img src="/img/goals.png" alt="Goals" className="rounded-lg w-3/4" />
//           </div>
//           <div className="md:w-1/2 mt-8 md:mt-0">
//             <h2 className="text-3xl font-bold text-black dark:text-white">{t("Our Goals")}</h2>
//             <p className="mt-4 text-black dark:text-white">{t("p4")}</p>
//           </div>
//         </section>

//         <Tooglebutton onValueChange={handleValueChange} />
//         <Footer />
//       </div>
//     </div>
//   );
// };

// export default About;
