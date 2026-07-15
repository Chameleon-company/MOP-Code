"use client";
import React, { useState, useEffect } from "react";
import { Link } from "@/i18n-navigation";

interface Blog {
  id: number;
  title: string;
  description: string | null;
  cover_img: string | null;
  published_date: string | null;
}

const BlogPage: React.FC = () => {
  const [blogs, setBlogs] = useState<Blog[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/home/blogs?page=1&pageSize=3")
      .then((r) => r.json())
      .then((json) => { if (json.success) setBlogs(json.data ?? []); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <section className="w-full bg-white dark:bg-[#1C1C1C] py-12 px-6 text-black dark:text-white">
      <div className="text-center mb-10">
        <h2 className="text-3xl md:text-4xl font-bold mb-3">
          Latest Blog Posts
        </h2>
        <p className="text-gray-600 dark:text-gray-400 text-sm md:text-base max-w-2xl mx-auto">
          Insights, updates, and expert tips to help you stay ahead in the digital world.
        </p>
      </div>

      {loading ? (
        <div className="flex justify-center py-10">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
        </div>
      ) : blogs.length === 0 ? (
        <div className="flex min-h-[160px] items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-gray-50 px-6 py-10 text-center dark:border-gray-700 dark:bg-[#242424]">
          <p className="text-base font-medium text-gray-500 dark:text-gray-400">
            No blog posts available at the moment.
          </p>
        </div>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 max-w-7xl mx-auto">
          {blogs.map((blog) => (
            <Link
              key={blog.id}
              href={`/blog/${blog.id}`}
              className="group block overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md dark:border-gray-700 dark:bg-gray-800"
            >
              {blog.cover_img && (
                <div className="h-44 overflow-hidden">
                  <img
                    src={blog.cover_img}
                    alt={blog.title}
                    className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                  />
                </div>
              )}
              <div className="p-5">
                <h3 className="mb-2 font-bold text-gray-900 dark:text-white line-clamp-2">{blog.title}</h3>
                {blog.description && (
                  <p className="text-sm text-gray-500 dark:text-gray-400 line-clamp-2">{blog.description}</p>
                )}
              </div>
            </Link>
          ))}
        </div>
      )}
    </section>
  );
};

export default BlogPage;