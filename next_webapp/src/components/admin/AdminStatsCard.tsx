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
    <div className="w-[142px] rounded-2xl bg-[#ECEAEA] px-4 py-5 text-center">
      <div className="mb-3 flex justify-center text-[#2DBE6C]">{icon}</div>
      <h3 className="text-[32px] font-semibold leading-none text-black">{value}</h3>
      <p className="mt-2 text-[14px] text-black">{title}</p>
    </div>
  );
}