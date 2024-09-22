import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";
import "../../../../public/styles/privacy.css";

const Privacypolicy: React.FC = () => {
  const t = useTranslations("privacypolicy");

  return (

    <div className="bg-gray-300"> {/* Slightly darker background for the page */}
      <Header />
      <main>
        <div className="h-full ms-auto content-center font-montserrat">
          <div className="ml-[5%] mt-[5%]">
            <h1 className="font-bold text-title">{t("Privacy Policy")}</h1> {/* Bold heading */}
          </div>
          <section className="md:flex md:flex-row">
            <div className="mx-[5%] mt-[6rem] md:mt-[8rem] justify-self-auto md:w-[40%]">
              <h2 className="text-[20px] font-semibold">{t("t1")}</h2>
              <div>
                <p className="mt-[2rem] text-[16px]">{t("p1")}</p>
              </div>
            </div>
            <div className="mx-[5%] mt-[4rem] md:mt-[8rem] justify-self-auto md:w-[40%]">
              <h2 className="text-[20px] font-semibold">{t("t2")}</h2>
              <div>
                <p className="mt-[2rem] text-[16px]">{t("p2")}</p>
              </div>
            </div>
          </section>
          <section className="md:flex md:flex-row">
            <div className="mx-[5%] mt-[4rem] justify-self-auto md:w-[40%]">
              <h2 className="text-[20px] font-semibold">{t("t3")}</h2>
              <div>
                <p className="mt-[2rem] text-[16px]">{t("p3")}</p>
              </div>
            </div>
            <div className="mx-[5%] mt-[4rem] justify-self-auto md:w-[40%]">
              <h2 className="text-[20px] font-semibold">{t("t4")}</h2>
              <div>
                <p className="mt-[2rem] text-[16px]">{t("p4")}</p>
              </div>
            </div>
          </section>
          <section className="md:flex md:flex-row">
            <div className="mx-[5%] mt-[4rem] justify-self-auto md:w-[40%]">
              <h2 className="text-[20px] font-semibold">{t("t5")}</h2>
              <div>
                <p className="mt-[2rem] text-[16px]">{t("p5")}</p>
              </div>
            </div>
            <div className="mx-[5%] mt-[4rem] justify-self-auto md:w-[40%]">
              <h2 className="text-[20px] font-semibold">{t("t6")}</h2>
              <div>
                <p className="mt-[2rem] text-[16px]">{t("p6")}</p>
              </div>
            </div>
          </section>
        </div>
        <div className="flex items-center justify-center mt-24 mb-[-3rem]">
            <p className="text-center text-[14px] w-[80%]">{t("p7")}</p>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Privacypolicy;

