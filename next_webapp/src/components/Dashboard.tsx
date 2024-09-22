"use client"
import { Link } from "@/i18n-navigation";
import Image from "next/image";
import mainimage from "../../public/img/mainImage.png";
import secondimage from "../../public/img/second_image.png";
import { useTranslations } from "next-intl";
import { CaseStudy, CATEGORY, SEARCH_MODE, SearchParams } from "@/app/types";
import { useEffect, useState } from "react";
import { ArrowLeft, FileText } from "lucide-react";

const style = `
@import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap');
.nav-menu{
    display: flex;
    flex-direction: row;
    gap: 29px;
}
body{
    background: white;
}

.nav-menu a{
    text-decoration: none;
    color: white;
}
.nav-section{
    background: #2ECC71;
    display: flex;
    justify-content: space-between;
    padding-top: 15px;
    padding-bottom: 15px;
    padding-left: 61px;
    padding-right: 24px;
    align-items: center;
    
}
.left-section{
    display: flex;
    gap: 68px;
    align-items: center;
}
.left-section .logo{
    height: 50px;
    width: 56px;
}
.left-section .logo img{
    width: 100%;
    height: 100%;
}
.sign-up{
    border: 1px solid white;
    text-decoration: none;
    padding: 12px 24px;
    background: #2ECC71;
    color: white;
    border-radius: 14px;
}
.log-in{
    color: #2ECC71;
    background: white;
    padding: 12px 24px;
    border: 1px solid white;
    border-radius: 14px;
}
.nav-section .right-section{
    display: flex;
    gap: 27px;
}
/* ------- */
.hero-section{
    margin-top: 55px;
    height: auto;
    background: grey;
    overflow: hidden;
}
.hero-section img{
    width: 100%;
    height : 805px;
}
.our-vision-section{
    display: flex;
    background: #2ECC71;
    justify-content: space-between;
    border: 1px solid white;
    margin-top: 21px;
    .our-vision{
        margin-left: 61px;
        margin-top: 98px;
        line-height: 60px;
        font-family: Montserrat;
        font-size: 64px;
        font-weight: 600;
        letter-spacing: -0.015em;
        text-align: left;
        color: white;
        background: #2ECC71;
    }
    .img-div{
        margin-top: 139px;
    max-width: 705px;
    width: 100%;
    border: 1px solid black;
    height: 429px;
    margin-left: 0px;
    margin-right: 25px;
    position: relative;
    }
    .img-div img{
        width: 100%;
        height:100%;
    }
    .img-div:after{
        content: '';
        width: 342px;
        height:55px;
        position: absolute;
        top: -56px;
        right: 0;
        background-color: #000000;
    }
    .img-div:before{
        content: '';
        width: 45px;
        height: 243px;
        position: absolute;
        background-color: #000000;
        left: -46px;
        bottom: -1px;
    }
    
   
    .text-div{
        margin-top: 284px;
        max-width: 686px;
        width: 100%;
        font-family: 'Montserrat';
        font-size: 17px;
        text-align: left;
        color: white;
    }
}
.case-studies-wrapper{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 85px;
}
.card-wrapper{
    border: 1px solid black;
    margin-top: 20px;
    width: 429px;
    height: 560px;
}
.card-wrapper h4{
    font-family: Montserrat;
font-size: 24px;
font-weight: 700;
margin-top: 47px;
margin-left: 64px;
color:black;
}
.card-wrapper p{
    font-family: Montserrat;
    font-size: 18px;
    font-weight: 400;
    margin-top: 24px;
    margin-left: 64px;
    color:black;
    max-width: 300px;
    width: 100%;
}
.card-wrapper .top-image{
    height: 281px;
    padding-left: 64px;
    padding-right:64px;
    padding-top:64px;
}
.card-wrapper .top-image img{
    width: 100%;
    height: 100%;
}
.recent-case-studies{
    margin-top: 105px;
    margin-left: 80px;
    margin-bottom: 82px;
}
.recent-case-studies h2{
    color : black;  
    font-family: Montserrat;
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 71px;
}
.recent-case-studies p{
    font-family: Montserrat;
    font-size: 36px;
    font-weight: 400;
    color : black;
}
.sign-up-btn-section{
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 16px;
    margin-bottom: 21px;
}
.sign-up-btn-section .sign-up-btn{
    width: 192px;
    height: 52px;
    color: white;
    font-size: 15px;
    font-weight: 500;
    background: #2ECC71;
    border: 0px;
    border-radius: 10px;
}
.main-wrapper{
    background-color: white;
}
`;
const Dashboard = () => {
    const [filteredCaseStudies, setFilteredCaseStudies] = useState<CaseStudy[]>([]);
    const [selectedCaseStudy, setSelectedCaseStudy] = useState<CaseStudy | undefined>(undefined)

    useEffect(() => {
        handleSearch("", SEARCH_MODE.TITLE, CATEGORY.ALL)
    }, [])

    async function searchUseCases(searchParams: SearchParams) {
        const response = await fetch("/api/search-use-cases", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(searchParams),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }
    const handleSearch = async (
        searchTerm: string,
        searchMode: SEARCH_MODE,
        category: CATEGORY
    ) => {
        const res = await searchUseCases({ searchTerm, searchMode, category });
        console.log("🚀 ~ UseCases ~ res:", res);
        setFilteredCaseStudies(res?.filteredStudies);
    };
    const handleCaseStudyClick = (study: CaseStudy) => {
        setSelectedCaseStudy(study);
    };
    const handleBack = () => {
        setSelectedCaseStudy(undefined);
    };
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

      <div className="main-wrapper dark:bg-zinc-800">
        <div className="main-container">
          <section className="hero-section">
            <Image src={mainimage} alt={"main image1"} />
          </section>
          <section className="sign-up-btn-section">
            <button className="sign-up-btn">
              <Link href="signup">{t("Sign Up")}</Link>
            </button>
          </section>
          <section className="our-vision-section">
            <div className="our-vision">{t("Our Vision")}</div>
            <div className="img-div">
              <Image src={secondimage} alt={"Second Image"} />
            </div>
            <div className="text-div">{t("intro")}</div>
          </section>
          <section className="recent-case-studies ">
            <h2 className="dark:text-slate-100">{t("Recent Case Studies")}</h2>
            <p className="dark:text-slate-100">{t("p2")}</p>
          </section>
    const t = useTranslations("common");

    if (selectedCaseStudy) {
        return (
            <div className="flex flex-col h-screen bg-gray-100 p-8">
                <button
                    onClick={handleBack}
                    className="flex items-center text-green-500 mb-4 hover:text-green-700 transition-colors duration-300"
                >
                    <ArrowLeft size={24} className="mr-2" />
                    Back
                </button>
                <div className="bg-white rounded-lg shadow-md p-6 flex-grow overflow-hidden">
                    <h1 className="text-3xl font-bold mb-4">{selectedCaseStudy.name}</h1>
                    <iframe
                        src={`/api?filename=${selectedCaseStudy.filename}`}
                        className="w-full h-full border-none"
                        title={selectedCaseStudy.name}
                    />
                </div>
            </div>
        );
    }

    return (
        <>
            <style dangerouslySetInnerHTML={{ __html: style }} />

            <div className="main-wrapper">
                <div className="main-container">
                    <section className="hero-section">
                        <Image src={mainimage} alt={"main image1"} />
                    </section>
                    <section className="sign-up-btn-section">
                        <button className="sign-up-btn">
                            <Link href="signup">{t("Sign Up")}</Link>
                        </button>
                    </section>
                    <section className="our-vision-section">
                        <div className="our-vision">{t("Our Vision")}</div>
                        <div className="img-div">
                            <Image src={secondimage} alt={"Second Image"} />
                        </div>
                        <div className="text-div">{t("intro")}</div>
                    </section>
                    <section className="recent-case-studies">
                        <h2>{t("Recent Case Studies")}</h2>
                        <p>{t("p2")}</p>
                    </section>

                    <section className="case-studies mx-10">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {filteredCaseStudies.slice(0, 3).map((study) => (
                                <div
                                    key={study.id}
                                    className="bg-white p-4 rounded-lg border-4 shadow-md cursor-pointer hover:shadow-2xl transition-shadow duration-300"
                                    onClick={() => handleCaseStudyClick(study)}
                                >
                                    <div className="flex items-center justify-center mb-4">
                                        <FileText size={48} className="text-green-500" />
                                        <FileText size={48} className="text-teal-400 -ml-6" />
                                        <FileText size={48} className="text-green-700 -ml-6 rotate-6" />
                                    </div>
                                    <h3 className="font-bold text-lg text-center mb-2">{study.name}</h3>
                                    <p className="text-gray-600 text-sm text-center mb-2">{study.description}</p>
                                    <div className="flex flex-wrap justify-center gap-2">
                                        <p className="text-sm text-gray-500">Tags:</p>
                                        {study.tags.map((tag, index) => (
                                            <span
                                                key={index}
                                                className="bg-gray-200 text-gray-800 text-xs px-2 py-1 rounded-full"
                                            >
                                                {tag}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>

                    </section>
                </div>
            </div>
        </>
    );
};

export default Dashboard;