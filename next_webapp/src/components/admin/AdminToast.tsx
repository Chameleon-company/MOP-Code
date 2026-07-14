"use client";

type AdminToastProps = {
  message: string;
  type: "success" | "error";
  onClose?: () => void;
};

export default function AdminToast({ message, type, onClose }: AdminToastProps) {
  if (!message) return null;

  return (
    <div className="fixed right-6 top-6 z-[1000]">
      <div
        className={`flex min-w-[260px] items-center justify-between gap-4 rounded-2xl px-5 py-4 text-sm font-medium shadow-lg ${
          type === "success"
            ? "border border-green-200 bg-green-50 text-green-700"
            : "border border-red-200 bg-red-50 text-red-700"
        }`}
      >
        <span>{message}</span>

        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="text-lg leading-none"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
}