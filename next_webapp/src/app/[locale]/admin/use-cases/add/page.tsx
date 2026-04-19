"use client";

import { useRouter } from "next/navigation";
import UseCaseForm from "../components/UseCaseForm";

export default function AddUseCase() {
  const router = useRouter();

  const handleSubmit = (data: any) => {
    console.log("NEW:", data);

    router.push("/en/admin/use-cases");
  };

  return (
    <div className="rounded-2xl bg-white p-8">
      <h2 className="mb-6 text-xl font-semibold">
        Add Use Case
      </h2>

      <UseCaseForm onSubmit={handleSubmit} />
    </div>
  );
}