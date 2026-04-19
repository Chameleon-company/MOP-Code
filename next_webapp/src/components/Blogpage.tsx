"use client";
import React from "react";

const BlogPage: React.FC = () => {
  return (
    <section className="w-full bg-white dark:bg-[#1C1C1C] py-12 px-6 text-black dark:text-white">
      <div className="text-center">
        <h2 className="text-3xl md:text-4xl font-bold mb-3">
          Latest Blog Posts
        </h2>
        <p className="text-gray-600 dark:text-gray-400 text-sm md:text-base max-w-2xl mx-auto">
          Insights, updates, and expert tips to help you stay ahead in the
          digital world.
        </p>
      </div>
    </section>
  );
};

export default BlogPage;
