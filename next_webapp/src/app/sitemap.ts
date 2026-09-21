import type { MetadataRoute } from "next";
import mongoose from "mongoose";
import dbConnect from "@/lib/dbConnect";
import UseCase from "@/models/mongoose/UseCase";
import Blog from "@/models/mongoose/Blog";
import { getHreflangAlternates } from "@/lib/seo/hreflang";

const STATIC_ROUTES = [
  "/",
  "/about",
  "/blog",
  "/usecases",
  "/gallery",
  "/faq",
  "/contact",
  "/licensing",
  "/privacypolicy",
  "/ev-infrastructure",
];

type SitemapEntry = MetadataRoute.Sitemap[number];

function buildEntry(pathname: string, lastModified?: Date): SitemapEntry {
  const { canonical, languages } = getHreflangAlternates(pathname);
  return {
    url: canonical,
    lastModified: lastModified ?? new Date(),
    alternates: { languages },
  };
}

async function getBlogEntries(): Promise<SitemapEntry[]> {
  await dbConnect();

  const docs = await Blog.find({}, { _id: 1, updated_at: 1 }).lean();

  return docs.map((doc) =>
    buildEntry(
      `/blog/${(doc._id as mongoose.Types.ObjectId).toString()}`,
      doc.updated_at ? new Date(doc.updated_at) : undefined,
    ),
  );
}

async function getUseCaseEntries(): Promise<SitemapEntry[]> {
  await dbConnect();

  const docs = await UseCase.find({}, { _id: 1, updated_at: 1 }).lean();

  return docs.map((doc) =>
    buildEntry(
      `/usecases/${(doc._id as mongoose.Types.ObjectId).toString()}`,
      doc.updated_at ? new Date(doc.updated_at) : undefined,
    ),
  );
}

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const staticEntries = STATIC_ROUTES.map((route) => buildEntry(route));

  const [blogEntries, useCaseEntries] = await Promise.all([
    getBlogEntries().catch((error) => {
      console.error("[sitemap] failed to load blog entries:", error);
      return [];
    }),
    getUseCaseEntries().catch((error) => {
      console.error("[sitemap] failed to load use case entries:", error);
      return [];
    }),
  ]);

  return [...staticEntries, ...blogEntries, ...useCaseEntries];
}
