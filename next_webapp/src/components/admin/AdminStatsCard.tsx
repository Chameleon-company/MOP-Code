import { ReactNode } from "react";

export default function AdminStatCard({
  title,
  value,
  icon,
}: {
  title: string;
  value: string;
  icon: ReactNode;
}) {
  return (
    <div className="w-full min-w-0 rounded-2xl bg-[#ECEAEA] px-4 py-6 text-center shadow-sm sm:px-5 sm:py-7 md:px-6 md:py-8">
      <div className="mb-3 flex justify-center text-[#2DBE6C] [&>svg]:h-7 [&>svg]:w-7 sm:[&>svg]:h-8 sm:[&>svg]:w-8">
        {icon}
      </div>
      <h3 className="text-2xl font-semibold leading-none text-black sm:text-3xl">
        {value}
      </h3>
      <p className="mt-2 text-sm text-black sm:text-base">{title}</p>
    </div>
  );
}