"use client";
import Image from "next/image";

type ImageHoverPreviewProps = {
  src: string;
  alt: string;
};

export default function ImageHoverPreview({ src, alt }: ImageHoverPreviewProps) {
  return (
    <div className="group relative inline-block">
      <Image
        src={src}
        alt={alt}
        width={64}
        height={48}
        className="h-12 w-16 rounded-lg object-cover border border-gray-200"
        unoptimized
      />

      <div className="pointer-events-none absolute left-20 top-0 z-50 hidden w-56 rounded-2xl border border-gray-200 bg-white p-2 shadow-xl group-hover:block">
        <Image
          src={src}
          alt={alt}
          width={224}
          height={144}
          className="h-36 w-full rounded-xl object-cover"
          unoptimized
        />
        <p className="mt-2 whitespace-normal break-words text-xs leading-5 text-gray-700">
  {alt}
</p>
      </div>
    </div>
  );
}