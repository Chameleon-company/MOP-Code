import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/about.css";
import { useTranslations } from "next-intl";

const About = () => {
  const t = useTranslations("about");
  return (
    <div className="bg-white">
      <Header />

      <div className="about-heading">{t("About")}</div>
      <div className="about-heading">{t("Us")}</div>
      <div className="image-container">
        <img src="/img/mel.jpg" alt="About Us Image" />
      </div>

      <div className="banner">
        <h2>{t("About MOP")}</h2>
        <p>{t("p1")}</p>
      </div>

      <div className="m-4 bg-white">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mx-10 mt-10">
          <div
            style={{ backgroundColor: "#cccccc", color: "black" }}
            className="flex flex-col items-center p-4 h-full"
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
            style={{ backgroundColor: "#cccccc", color: "black" }}
            className="flex flex-col items-center p-4 h-full"
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
            style={{ backgroundColor: "#cccccc", color: "black" }}
            className="flex flex-col items-center p-4 h-full"
          >
            <div className="flex flex-row h-full">
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
  );
};

export default About;
