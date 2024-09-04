import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";

const Privacypolicy = () => {
  const t = useTranslations("privacypolicy");

  return (
    <div>
      <Header />
      <main>
        <div className="h-[70rem] px-[5rem] content-center font-sans-serif bg-white">
          <h1 className="pl-20 p-8 font-semibold text-4xl mt-32">
            {t("Privacy Policy")}
          </h1>
          <section className="  md:flex md:flex-row">
            <div className=" pl-[5rem] pt-[8rem] pr-[8rem] -mt-14 justify-self-auto md:w-1/2">
              <h2 className="text-[20px] font-normal font-semibold">
                {" "}
                {t("t1")}
              </h2>
              <div>
                <p className=" mt-[2rem] text-[16px] font-dark">{t("p1")}</p>
              </div>
            </div>
            <div className=" pl-[5rem] pt-[8rem] pr-[6rem] -mt-14 justify-self-auto md:w-1/2">
              <h2 className="text-[20px] font-normal font-semibold">
                {t("t2")}
              </h2>
              <div>
                <p className=" mt-[2rem] text-[16px] font-dark">{t("p2")}</p>
              </div>
            </div>
          </section>
          <section className="md:flex md:flex-row">
            <div className="pl-[5rem] pt-[4rem] pr-[8rem] justify-self-auto md:w-1/2">
              <h2 className="text-[20px] font-normal font-semibold">
                {t("t3")}{" "}
              </h2>
              <div>
                <p className=" mt-[2rem] text-[16px] font-dark"> {t("p3")}</p>
              </div>
            </div>
            <div className="pl-[5rem] pt-[4rem] pr-[6rem] justify-self-auto md:w-1/2">
              <h2 className="text-[20px] font-normal font-semibold">
                {" "}
                {t("t4")}
              </h2>
              <div>
                <p className=" mt-[2rem] text-[16px] font-dark">{t("p4")}</p>
              </div>
            </div>
          </section>
          <section className="md:flex md:flex-row">
            <div className="pl-[5rem] pt-[4rem] pr-[8rem] justify-self-auto md:w-1/2">
              <h2 className=" text-[20px] font-normal font-semibold">
                {t("t5")}{" "}
              </h2>
              <div>
                <p className=" mt-[2rem] text-[16px] font-dark"> {t("p5")}</p>
              </div>
            </div>
            <div className="pl-[5rem] pt-[4rem] pr-[6rem] justify-self-auto md:w-1/2">
              <h2 className="text-[20px] font-normal font-semibold">
                {" "}
                {t("t6")}
              </h2>
              <div>
                <p className=" mt-[2rem] text-[16px] font-dark">{t("p6")}</p>
              </div>
            </div>
          </section>
          <div className="flex items-center justify-center p-3 mt-8 mb-6">
            <p className="text-center">{t("p7")}</p>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

//privacy page has been fixed

export default Privacypolicy;