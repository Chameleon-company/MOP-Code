import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";
import "../../../../public/styles/privacy.css";

const Privacypolicy: React.FC = () => {
  const t = useTranslations("privacypolicy");

  const sections = [
    { title: t("t1"), content: t("p1") },
    { title: t("t2"), content: t("p2") },
    { title: t("t3"), content: t("p3") },
    { title: t("t4"), content: t("p4") },
    { title: t("t5"), content: t("p5") },
    { title: t("t6"), content: t("p6") },
  ];

  return (
    <div className="bg-gray-300 min-h-screen font-montserrat">
      <Header />

      <main className="px-6 py-12 md:px-[5%]">
        <h1 className="text-title font-bold text-3xl mb-10">{t("Privacy Policy")}</h1>

        <div className="space-y-16">
          {/* Loop through sections in pairs */}
          {[0, 2, 4].map((i) => (
            <section key={i} className="md:flex md:justify-between md:space-x-8">
              <div className="md:w-1/2 mb-12 md:mb-0">
                <h2 className="text-[20px] font-semibold">{sections[i].title}</h2>
                <p className="mt-6 text-[16px]">{sections[i].content}</p>
              </div>
              <div className="md:w-1/2">
                <h2 className="text-[20px] font-semibold">{sections[i + 1].title}</h2>
                <p className="mt-6 text-[16px]">{sections[i + 1].content}</p>
              </div>
            </section>
          ))}
        </div>

        <div className="flex items-center justify-center mt-24">
          <p className="text-center text-[14px] max-w-4xl">{t("p7")}</p>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Privacypolicy;
