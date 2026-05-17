"use client";

type ImageHoverPreviewProps = {
  src: string;
  alt: string;
};

export default function ImageHoverPreview({ src, alt }: ImageHoverPreviewProps) {
  return (
    <div className="group relative inline-block">
      <img
        src={src}
        alt={alt}
        className="h-12 w-16 rounded-lg object-cover border border-gray-200"
      />

      <div className="pointer-events-none absolute left-20 top-0 z-50 hidden w-56 rounded-2xl border border-gray-200 bg-white p-2 shadow-xl group-hover:block">
        <img
          src={src}
          alt={alt}
          className="h-36 w-full rounded-xl object-cover"
        />
        <p className="mt-2 whitespace-normal break-words text-xs leading-5 text-gray-700">
  {alt}
</p>
      </div>
    </div>
  );
}