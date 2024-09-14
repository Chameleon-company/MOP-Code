import { Link } from "@/i18n-navigation";
import Image from "next/image";
import mainimage from "../../public/img/mainImage.png";
import secondimage from "../../public/img/second_image.png";
import { useTranslations } from "next-intl";

const Dashboard = () => {
  const navItems = [
    { to: "/about", icon: "/img/about-icon.png", label: "About Us" },
    { to: "/casestudies", icon: "/img/case-icon.png", label: "Case Studies" },
    {
      to: "/resource-center",
      icon: "/img/resource-icon.png",
      label: "Resource Center",
    },
    { to: "/datasets", icon: "/img/data-icon.png", label: "Data Collection" },
    { to: "/contact", icon: "/img/contact-icon.png", label: "Contact Us" },
  ];

  const t = useTranslations("common");

  return (
    <>
      <div className="bg-white">
        <div className="w-full">
          {/* Hero Section */}
          <section className="hero-section w-full h-auto mt-12">
            <Image src={mainimage} alt={"main image"} className="w-full h-auto" />
          </section>

          {/* Sign Up Button Section */}
          <section className="flex justify-center items-center mt-4 mb-5">
            <button className="bg-green-500 text-white font-medium py-3 px-6 rounded-lg">
              <Link href="signup">{t("Sign Up")}</Link>
            </button>
          </section>

          {/* Our Vision Section */}
          <section className="bg-green-500 text-white py-8 px-4 md:px-16 lg:flex justify-between items-center">
            <div className="flex flex-col justify-center">
              <h1 className="text-3xl md:text-5xl font-bold mb-8">{t("Our Vision")}</h1>
              <p className="text-lg md:text-xl leading-relaxed">{t("intro")}</p>
            </div>
            <div className="relative mt-8 lg:mt-0 lg:w-1/2">
              <Image src={secondimage} alt={"Second Image"} className="w-full h-auto" />
              <div className="absolute top-0 right-0 w-16 h-16 bg-black"></div>
              <div className="absolute bottom-0 left-0 w-8 h-40 bg-black"></div>
            </div>
          </section>

          {/* Recent Case Studies Section */}
          <section className="recent-case-studies  mt-20 mb-20 px-4 md:px-16">
            <h2 className="text-black text-4xl font-bold mb-12">{t("Recent Case Studies")}</h2>
            <p className="text-3xl  text-black">{t("p2")}</p>
          </section>
        </div>
      </div>
    </>
  );
};

export default Dashboard;