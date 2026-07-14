export default function Loading() {
  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-white dark:bg-[#1C1C1C]">
      <div className="flex flex-col items-center gap-4">
        <div className="h-12 w-12 animate-spin rounded-full border-4 border-green-200 border-t-green-600 dark:border-green-800 dark:border-t-green-400" />
        <p className="text-sm font-semibold text-green-600 dark:text-green-400">Loading...</p>
      </div>
    </div>
  );
}
