const activityData = [
  { date: "02/04/2026 07:00", activity: "Blog Added", status: "Published" },
  { date: "01/04/2026 16:00", activity: "New category created", status: "Published" },
  { date: "01/04/2026 12:00", activity: "New category created", status: "Pending" },
];

export default function AdminRecentActivity() {
  return (
    <div className="mt-10">
      <h2 className="mb-4 text-[28px] font-semibold leading-[36px] text-[#2DBE6C]">
        Recent Activity
      </h2>

      <div className="rounded-2xl bg-[#ECEAEA] p-4">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-black">
              <th className="px-2 py-3 text-left text-[14px] font-semibold text-black">Date</th>
              <th className="px-2 py-3 text-left text-[14px] font-semibold text-black">Activity</th>
              <th className="px-2 py-3 text-left text-[14px] font-semibold text-black">Status</th>
            </tr>
          </thead>
          <tbody>
            {activityData.map((item, index) => (
              <tr key={index}>
                <td className="px-2 py-3 text-[14px] text-black">{item.date}</td>
                <td className="px-2 py-3 text-[14px] text-black">{item.activity}</td>
                <td className="px-2 py-3 text-[14px] text-black">{item.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}