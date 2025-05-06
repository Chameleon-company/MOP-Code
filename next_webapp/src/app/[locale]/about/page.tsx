// "use client";
// import Header from "../../../components/Header";
// import Footer from "../../../components/Footer";
// import "../../../../public/styles/about.css";
// import { useTranslations } from "next-intl";
// import { useState } from "react";
// import Tooglebutton from "../Tooglebutton/Tooglebutton";

// const About = () => {
//   const t = useTranslations("about");

//   //dark theme
//   const [dark_value, setdarkvalue] = useState(false);
//   const handleValueChange = (
//     newValue: boolean | ((prevState: boolean) => boolean)
//   ) => {
//     setdarkvalue(newValue);
//   };
//   return (
//     <div className={`${dark_value && "dark"}`}>
//       <div className="bg-white dark:bg-[#1d1919]">
//         <Header />
//         <div className="h-full content-center">
//           <div className="container-title bg-green-500 dark:bg-green-900">
//             <div className="content-title md:py-8">
//               <img
//                 src="/img/mel.jpg"
//                 alt="Melbourne Open Playground"
//                 className="float-left img-section pr-[5%]"
//               />
//               <div className="text-box font-sans">
//                 <h2 className="font-bold">{t("About Us")}</h2>
//                 <br />
//                 <p className="w-full md:w-[60%]" id="p2">
//                   {t("p2")}
//                 </p>
//               </div>
//             </div>
//           </div>

//           <div className="container bg-white dark:bg-[#1d1919]">
//             <div className="content-section md:pt-16">
//               <img
//                 src="/img/leadership.png"
//                 alt="Leadership Image"
//                 className="float-left img-section pr-[5%]"
//               />
//               <div className="text-box font-sans">
//                 <h2 className="font-bold text-black dark:text-white">
//                   {t("Open Data Leadership")}
//                 </h2>
//                 <br />
//                 <p
//                   className="text-black dark:text-white w-full md:w-[60%]"
//                   id="p3"
//                 >
//                   {t("p3")}
//                 </p>
//               </div>
//             </div>
//           </div>

//           <div className="container bg-white  dark:bg-[#1d1919]">
//             <div className="content-section md:pt-6">
//               <img
//                 src="/img/goals.png"
//                 alt="Our Goals"
//                 className="float-left img-section pr-[5%]"
//               />
//               <div className="text-box font-sans">
//                 <h2 className="font-bold text-black dark:text-white">
//                   {t("Our Goals")}
//                 </h2>
//                 <br />
//                 <p className="text-black dark:text-white" id="p4">
//                   {t("p4")}
//                 </p>
//               </div>
//             </div>
//           </div>
//         </div>

//         <Tooglebutton onValueChange={handleValueChange} />
//         <Footer />
//       </div>
//     </div>
//   );
// };

// export default About;

//Divyanga C.S.Lokuhetti #s223590519
//Team Project(B) - T1 2025
"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/about.css";
import { useTranslations } from "next-intl";
import { useState, useEffect } from "react";
import Tooglebutton from "../Tooglebutton/Tooglebutton";

const About = () => {
  const t = useTranslations("about");

  const [dark_value, setdarkvalue] = useState(false);

  const handleValueChange = (
    newValue: boolean | ((prevState: boolean) => boolean)
  ) => {
    setdarkvalue(typeof newValue === "function" ? newValue(dark_value) : newValue);
  };

  // Apply/remove dark mode class from <html> or <body>
  useEffect(() => {
    const root = document.documentElement;
    if (dark_value) {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
  }, [dark_value]);

  return (
    <div className="bg-white dark:bg-[#1d1919] min-h-screen">
      <Header />
      <div className="h-full content-center">
        {/* Section 1 */}
        <div className="container-title bg-green-500 dark:bg-green-900">
          <div className="content-title md:py-8">
            <img
              src="/img/mel.jpg"
              alt="Melbourne Open Playground"
              className="float-left img-section pr-[5%]"
            />
            <div className="text-box font-sans">
              <h2 className="font-bold text-black dark:text-white">{t("About Us")}</h2>
              <br />
              <p className="text-black dark:text-white w-full md:w-[60%]" id="p2">
                {t("p2")}
              </p>
            </div>
          </div>
        </div>

        {/* Section 2 */}
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
              <p className="text-black dark:text-white w-full md:w-[60%]" id="p3">
                {t("p3")}
              </p>
            </div>
          </div>
        </div>

        {/* Section 3 */}
        <div className="container bg-white dark:bg-[#1d1919]">
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

      {/* Dark Mode Toggle */}
      <div className="fixed bottom-4 right-4 z-50">
        <Tooglebutton onValueChange={handleValueChange} />
      </div>

      <Footer />
    </div>
  );
};

export default About;
