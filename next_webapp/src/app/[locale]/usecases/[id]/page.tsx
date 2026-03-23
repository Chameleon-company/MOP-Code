"use client";
import React from "react";
import { useRouter } from "next/router";
import Link from "next/link";
import { useParams } from "next/navigation";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { useCases } from "@/utils/data";
// import your updated array

const UseCasePage: React.FC = () => {
  const params = useParams();
  const id = params.id;
  console.log("The id : ", id);

  const useCase = useCases.find((uc: any) => uc.id === id);

  if (!useCase) return <p className="text-center py-20">Use case not found.</p>;

  return (
    <>
      <Header />
      <section className="max-w-6xl mx-auto py-12 px-6 bg-white dark:bg-[#263238] text-black dark:text-white min-h-screen">
        {/* Main Use Case */}
        <div className="mb-12 text-center">
          <h1 className="text-4xl font-bold mb-4">{useCase.title}</h1>
          <p className="text-gray-600 dark:text-gray-300 text-lg max-w-3xl mx-auto">
            {useCase.description}
          </p>
          <img
            src={useCase.image}
            alt={useCase.title}
            className="rounded-xl mt-6 mx-auto w-full max-w-3xl h-80 object-cover shadow-lg"
          />
        </div>
        <p className="text-black font-bold text-center mb-5 mt-5 dark:text-gray-300 text-xl max-w-3xl mx-auto">
          Sub-Usecases
        </p>
        {/* Sub Use Cases */}
        <div className="grid gap-10">
          {useCase.subUseCases.map((sub, index) => (
            <div
              key={index}
              className="flex flex-col md:flex-row items-center gap-6 bg-gray-50 dark:bg-[#37474F] p-6 rounded-2xl shadow-md"
            >
              <img
                src={sub.image}
                alt={sub.heading}
                className="w-full md:w-1/2 h-60 object-cover rounded-xl shadow-lg"
              />
              <div className="md:w-1/2">
                <h2 className="text-2xl font-semibold mb-3 text-gray-800 dark:text-white">
                  {sub.heading}
                </h2>
                <p className="text-gray-700 dark:text-gray-300">
                  {sub.description}
                </p>
                <Link
                  href={`${sub.link}`}
                  className="mt-4 inline-block bg-green-500 hover:bg-green-600 text-white py-2 px-4 rounded-xl text-sm font-medium text-center self-start"
                >
                  View →
                </Link>
              </div>
            </div>
          ))}
        </div>
      </section>
      <Footer />
    </>
  );
};

export default UseCasePage;
