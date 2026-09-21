"use client";
import Image from "next/image";
import { useEffect, useState } from "react";

const FALLBACK_IMAGE = "/images/category-placeholder.png";

type ImageHoverPreviewProps = {
  src: string;
  alt: string;
};

export default function ImageHoverPreview({ src, alt }: ImageHoverPreviewProps) {
  const [imgSrc, setImgSrc] = useState(src || FALLBACK_IMAGE);

  useEffect(() => {
    setImgSrc(src || FALLBACK_IMAGE);
  }, [src]);

  return (
    <div className="group relative inline-block">
      <Image
        src={imgSrc}
        alt={alt}
        width={64}
        height={48}
        className="h-12 w-16 rounded-lg object-cover border border-gray-200"
        onError={() => setImgSrc(FALLBACK_IMAGE)}
      />

      <div className="pointer-events-none absolute left-20 top-0 z-50 hidden w-56 rounded-2xl border border-gray-200 bg-white p-2 shadow-xl group-hover:block">
        <Image
          src={imgSrc}
          alt={alt}
          width={224}
          height={144}
          className="h-36 w-full rounded-xl object-cover"
          onError={() => setImgSrc(FALLBACK_IMAGE)}
        />
        <p className="mt-2 whitespace-normal break-words text-xs leading-5 text-gray-700">
  {alt}
</p>
      </div>
    </div>
  );
}