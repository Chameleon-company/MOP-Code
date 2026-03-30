"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";

const Gallery = () => {
  return (
    <div className="min-h-screen flex flex-col bg-white dark:bg-black">
      <Header />
      <main className="flex-1 flex flex-col items-center justify-center px-4 py-24">
        <h1 className="text-4xl font-bold text-gray-800 dark:text-white mb-4">
          Gallery
        </h1>
        <p className="text-lg text-gray-500 dark:text-gray-400">
          Coming Soon
        </p>
      </main>
      <Footer />
    </div>
  );
};

export default Gallery;
