import BlogPage from "@/components/Blogpage";
import Footer from "@/components/Footer";
import Header from "@/components/Header";
import React from "react";

function page() {
  return (
    <div className="flex-1 flex flex-col">
      <Header />
      <div className="flex-1">
        <BlogPage />
      </div>
      <Footer />
    </div>
  );
}

export default page;
