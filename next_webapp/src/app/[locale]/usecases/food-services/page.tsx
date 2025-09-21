"use client";

import Header from "../../../../components/Header";
import Footer from "../../../../components/Footer";
import FoodServicesCaseBody from "../../../../components/FoodServicesCaseBody";

export default function FoodServicesPage() {
  return (
    <div className="flex flex-col min-h-screen font-sans bg-white dark:bg-gray-800 text-black dark:text-white transition-all duration-300">
      {/* Header */}
      <Header />

      {/* Main content */}
      <main className="flex-grow bg-[#0e1621]">
        <FoodServicesCaseBody />
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
}
