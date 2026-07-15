"use client";

type TextHoverPreviewProps = {
  text: string;
};

export default function TextHoverPreview({ text }: TextHoverPreviewProps) {
  return (
    <div className="group relative max-w-[260px]">
      <p className="line-clamp-2 cursor-default text-sm text-[#687280]">
        {text}
      </p>

      <div className="pointer-events-none absolute left-0 top-full z-50 mt-2 hidden w-80 rounded-xl border border-[#CFEFD9] bg-white p-3 text-sm text-black shadow-xl group-hover:block">
        {text}
      </div>
    </div>
  );
}