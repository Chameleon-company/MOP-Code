"use client";
import React from "react";
import { Link } from "@/i18n-navigation";
import { blogs } from "@/utils/data";

const BlogPage: React.FC = () => {
  return (
    <section className="w-full bg-white dark:bg-[#1C1C1C] py-12 px-6 text-black dark:text-white">
      {/* Header */}
      <div className="text-center mb-10">
        <h2 className="text-3xl md:text-4xl font-bold mb-3">
          Latest Blog Posts
        </h2>
        <p className="text-gray-600 dark:text-gray-400 text-sm md:text-base max-w-2xl mx-auto">
          Insights, updates, and expert tips to help you stay ahead in the
          digital world.
        </p>
      </div>

      {/* Blog Grid */}
      <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
        {blogs.map((blog) => (
          <div
            key={blog.id}
            className="bg-gray-50 dark:bg-[#2A2A2A] rounded-2xl shadow-md hover:shadow-xl transition flex flex-col group overflow-hidden"
          >
            {/* Blog Image */}
            <div className="relative w-full h-48">
              <img
                src={blog.image}
                alt={blog.title}
                className="w-full h-full object-cover group-hover:scale-[1.05] transition-transform"
              />
            </div>

            {/* Blog Content */}
            <div className="p-5 flex flex-col flex-grow">
              <span className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                {blog.date} • {blog.author}
              </span>
              <h3 className="text-lg font-semibold mb-2 group-hover:text-green-500 transition-colors">
                {blog.title}
              </h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm flex-grow">
                {blog.description}
              </p>
              <Link
                href={`/blog/${blog.id}`}
                className="mt-4 inline-block bg-green-500 hover:bg-green-600 text-white py-2 px-4 rounded-xl text-sm font-medium text-center self-start"
              >
                Read More →
              </Link>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

export default BlogPage;
