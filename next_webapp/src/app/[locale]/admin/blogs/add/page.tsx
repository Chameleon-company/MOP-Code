"use client";

import { useRouter } from "next/navigation";
import BlogForm from "../components/BlogsForm";

export default function AddBlog() {
	const router = useRouter();

	const handleSubmit = (data: any) => {
		console.log("New blog:", data);

		router.push("/en/admin/blogs");
	};

	return (
		<div className="rounded-2xl bg-white p-8">
			<h2 className="mb-6 text-xl font-semibold">Add Blog</h2>

			<BlogForm onSubmit={handleSubmit} />
		</div>
	);
}
