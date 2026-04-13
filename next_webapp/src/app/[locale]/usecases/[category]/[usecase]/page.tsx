"use client";

import React, { useEffect, useState } from "react";
import { useParams, notFound } from "next/navigation";
import Link from "next/link";
import Header from "../../../../../components/Header";
import Footer from "../../../../../components/Footer";
import usecasesData from "../../../../utils/usecases.json";
import { useLocale } from "next-intl";

interface UseCase {
    name: string;
    address: string; // directory only
}
interface Category {
    category: string;
    usecases: UseCase[];
}

function slugify(text: string) {
    return text
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/(^-|-$)+/g, "");
}

export default function UseCasePage() {
    const locale = useLocale();
    const params = useParams<{ category: string; usecase: string }>();
    const { category, usecase } = params;

    const [content, setContent] = useState<string | null>(null);

    // Find matching category and usecase
    const cat = (usecasesData as Category[]).find(
        (c) => slugify(c.category) === category
    );
    const uc = cat?.usecases.find((u) => slugify(u.name) === usecase);

    useEffect(() => {
        async function fetchFile() {
            if (!uc) return;

            try {
                const parts = uc.address.split("/").filter(Boolean);
                const last = parts[parts.length - 1];
                const htmlPath = `${uc.address}${last}.html`.replace(/\/+/, "/");

                const res = await fetch(htmlPath);
                if (!res.ok) throw new Error("Not found");

                setContent(htmlPath);
            } catch (err) {
                console.error(err);
                setContent(null);
            }
        }

        fetchFile();
    }, [uc]);

    if (!uc) {
        notFound();
    }

    return (
        <div className="flex flex-col min-h-screen font-sans bg-white dark:bg-gray-800 text-black dark:text-white transition-all duration-300">
            <Header />

            <main className="flex-grow w-full max-w-7xl mx-auto px-4 py-8">
                {/* Breadcrumbs */}
                <nav className="mb-6 text-sm text-gray-600 dark:text-gray-300">
                    <Link href="/" className="hover:underline">Home</Link> &gt;{" "}
                    <Link href={`/${locale}/usecases`} className="hover:underline">Use Cases</Link> &gt;{" "}
                    <Link href={`/${locale}/usecases/${category}`} className="hover:underline">
                        {cat?.category}
                    </Link> &gt;{" "}
                    <span className="font-semibold">{uc?.name}</span>
                </nav>

                {/* Title above iframe */}
                <h1 className="text-3xl font-bold mb-6 text-green-600">
                    {uc?.name}
                </h1>

                {content && (
                    <iframe
                        src={content}
                        className="w-full h-[85vh] border-0 rounded-xl shadow"
                        title={uc.name}
                    />
                )}
            </main>

            <Footer />
        </div>
    );
}
