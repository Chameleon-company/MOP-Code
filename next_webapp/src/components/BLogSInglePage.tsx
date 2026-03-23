"use client";
import React from "react";
import { useParams } from "next/navigation";
import { blogs } from "@/utils/data";

const BlogSinglePage: React.FC = () => {
  const { id } = useParams();
  const blog = blogs.find((b) => b.id.toString() === id);

  if (!blog) {
    return (
      <div className="flex items-center justify-center min-h-screen text-gray-600 dark:text-gray-300">
        <p>Blog post not found :)</p>
      </div>
    );
  }

  return (
    <article className="w-full min-h-screen bg-white dark:bg-[#1C1C1C] text-black dark:text-white px-6 md:px-12 py-12">
      <header className="max-w-4xl mx-auto text-center mb-10">
        <h1 className="text-3xl md:text-4xl font-bold mb-4">{blog.title}</h1>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
          {blog.date} • {blog.author}
        </p>
      </header>

      <div className="max-w-5xl mx-auto mb-8">
        <img
          src={blog.image}
          alt={blog.title}
          className="w-full h-64 md:h-96 object-cover rounded-2xl shadow-lg"
        />
      </div>

      {/* Blog Content */}
      <section className="max-w-3xl mx-auto text-gray-700 dark:text-gray-300 leading-relaxed space-y-6">
        <p>{blog.description}</p>

        <p>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ac nulla
          euismod, dictum arcu vel, placerat nisl. Nulla facilisi. Integer
          egestas libero a dui euismod, vitae bibendum felis vulputate.
        </p>
        <p>
          Praesent dictum sapien non justo iaculis, vitae tristique nibh
          tincidunt. Duis dapibus faucibus velit a ullamcorper. Integer ac nunc
          nec purus congue consequat.
        </p>

        <p>
          Praesent dictum sapien non justo iaculis, vitae tristique nibh
          tincidunt. Duis dapibus faucibus velit a ullamcorper. Integer ac nunc
          nec purus congue consequat.
        </p>
      </section>

      <footer className="max-w-3xl mx-auto mt-12 border-t border-gray-200 dark:border-gray-700 pt-6 text-sm text-gray-600 dark:text-gray-400">
        <p>
          Written by <span className="font-semibold">{blog.author}</span>
        </p>
      </footer>
    </article>
  );
};

export default BlogSinglePage;
