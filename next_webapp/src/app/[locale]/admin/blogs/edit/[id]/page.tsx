"use client";

import { useRouter, useParams } from "next/navigation";
import BlogForm from "../../components/BlogsForm";

export default function EditBlogPage() {
  const router = useRouter();
  const params = useParams();

  const initialData = {
    title: "How Data Supports Better City Planning",
    subTitle: "Using data to improve community planning",
    date: "2026-04-27",
    description:
      "Explore how data-driven insights help improve urban planning and community services.",
    image: "/images/category-placeholder.png",
  };

  const handleSubmit = (data: any) => {
    console.log("Updated blog:", data);

    router.push(`/${params.locale}/admin/blogs`);
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-[40px] font-semibold text-[#2DBE6C]">
          Edit Blog
        </h1>
        <p className="mt-2 text-[16px] text-[#687280]">
          Update blog details
        </p>
      </div>

      <div className="rounded-2xl bg-white p-8">
        <BlogForm initialData={initialData} onSubmit={handleSubmit} />
      </div>
    </div>
  );
}