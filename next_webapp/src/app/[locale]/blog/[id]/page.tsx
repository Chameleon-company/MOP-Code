import type { Metadata } from "next";
import mongoose from "mongoose";
import { getTranslations } from "next-intl/server";
import BlogSinglePage from "@/components/BLogSInglePage";
import Footer from "@/components/Footer";
import Header from "@/components/Header";
import dbConnect from "@/lib/dbConnect";
import Blog from "@/models/mongoose/Blog";
import { getHreflangAlternates } from "@/lib/seo/hreflang";

type BlogDetailParams = Promise<{ locale: string; id: string }>;

async function getBlogForMetadata(id: string) {
  if (!mongoose.Types.ObjectId.isValid(id)) return null;

  await dbConnect();

  const blog = await Blog.findById(id, {
    title: 1,
    description: 1,
    cover_img: 1,
  }).lean();

  return blog;
}

export async function generateMetadata({
  params,
}: {
  params: BlogDetailParams;
}): Promise<Metadata> {
  const { locale, id } = await params;
  const [t, blog] = await Promise.all([
    getTranslations({ locale, namespace: "seo" }),
    getBlogForMetadata(id),
  ]);

  const { canonical, languages } = getHreflangAlternates(`/blog/${id}`);

  if (!blog) {
    return {
      title: t("blog_title"),
      description: t("blog_description"),
      alternates: { canonical, languages },
    };
  }

  const title = blog.title as string;
  const description = (blog.description as string | null) ?? t("blog_description");
  const coverImg = blog.cover_img as string | null;

  return {
    title,
    description,
    alternates: { canonical, languages },
    openGraph: {
      type: "article",
      title,
      description,
      url: canonical,
      ...(coverImg ? { images: [{ url: coverImg }] } : {}),
    },
  };
}

export default async function BlogDetailPage({
  params,
}: {
  params: BlogDetailParams;
}) {
  const { id } = await params;

  return (
    <div>
      <Header />
      <BlogSinglePage id={id} />
      <Footer />
    </div>
  );
}
