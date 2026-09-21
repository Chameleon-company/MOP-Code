import type { Metadata } from "next";
import mongoose from "mongoose";
import { getTranslations } from "next-intl/server";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import dbConnect from "@/lib/dbConnect";
import UseCase from "@/models/mongoose/UseCase";
import { toUseCaseDTO } from "@/app/api/library/useCaseDto";
import { getHreflangAlternates } from "@/lib/seo/hreflang";
import UseCaseDetailClient from "./UseCaseDetailClient";

type UseCaseDetailParams = Promise<{ locale: string; id: string }>;

async function getUseCaseForMetadata(id: string) {
  if (!mongoose.Types.ObjectId.isValid(id)) return null;

  await dbConnect();
  const doc = await UseCase.findById(id).lean();
  if (!doc) return null;

  return toUseCaseDTO(doc);
}

export async function generateMetadata({
  params,
}: {
  params: UseCaseDetailParams;
}): Promise<Metadata> {
  const { locale, id } = await params;
  const [t, useCase] = await Promise.all([
    getTranslations({ locale, namespace: "seo" }),
    getUseCaseForMetadata(id),
  ]);

  const { canonical, languages } = getHreflangAlternates(`/usecases/${id}`);

  if (!useCase) {
    return {
      title: t("usecases_title"),
      description: t("usecases_description"),
      alternates: { canonical, languages },
    };
  }

  const title = useCase.title as string;
  const description = (useCase.description as string | null) ?? t("usecases_description");
  const coverImg = useCase.cover_img as string | null;

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

export default async function UseCaseDetailPage({
  params,
}: {
  params: UseCaseDetailParams;
}) {
  const { id } = await params;

  return (
    <>
      <Header />
      <UseCaseDetailClient id={id} />
      <Footer />
    </>
  );
}
