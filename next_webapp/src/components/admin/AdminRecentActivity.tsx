const activityData = [
  { date: "02/04/2026 07:00", activity: "Blog Added", status: "Published" },
  {
    date: "01/04/2026 16:00",
    activity: "New category created",
    status: "Published",
  },
  {
    date: "01/04/2026 12:00",
    activity: "New category created",
    status: "Pending",
  },
];

export default function AdminRecentActivity() {
  return (
    <div className="mt-8 md:mt-10">
      <h2 className="mb-4 text-2xl font-semibold leading-tight text-[#2DBE6C] sm:text-3xl md:text-[28px]">
        Recent Activity
      </h2>

      <div className="w-full overflow-x-auto rounded-2xl bg-[#ECEAEA] p-4 shadow-sm sm:p-5 md:p-6">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-black">
              <th className="px-2 py-3 text-left text-[13px] font-semibold text-black sm:text-[14px]">
                Date
              </th>
              <th className="px-2 py-3 text-left text-[13px] font-semibold text-black sm:text-[14px]">
                Activity
              </th>
              <th className="px-2 py-3 text-left text-[13px] font-semibold text-black sm:text-[14px]">
                Status
              </th>
            </tr>
          </thead>
          <tbody>
            {activityData.map((item, index) => (
              <tr
                key={index}
                className="border-b border-gray-300 last:border-b-0"
              >
                <td className="px-2 py-3 text-[13px] text-black sm:text-[14px]">
                  {item.date}
                </td>
                <td className="px-2 py-3 text-[13px] text-black sm:text-[14px]">
                  {item.activity}
                </td>
                <td className="px-2 py-3 text-[13px] text-black sm:text-[14px]">
                  {item.status}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}