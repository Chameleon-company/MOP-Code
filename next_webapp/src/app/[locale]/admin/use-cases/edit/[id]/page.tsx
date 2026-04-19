"use client";

import { useRouter } from "next/navigation";
import UseCaseForm from "../../components/UseCaseForm";

export default function EditUseCase({ params }: any) {
  const router = useRouter();

  // fake existing data
  const existingData = {
    serialNumber:"Existing Serial Number",
    title: "Existing Title",
    description: "Existing description",
    image: "/images/category-placeholder.png",
  };

  const handleSubmit = (data: any) => {
    console.log("UPDATED:", data);

    router.push("/en/admin/use-cases");
  };

  return (
    <div className="rounded-2xl bg-white p-8">

      <h2 className="mb-6 text-xl font-semibold">
        Edit Use Case
      </h2>

      <UseCaseForm
        initialData={existingData}
        onSubmit={handleSubmit}
      />

    </div>
  );
}