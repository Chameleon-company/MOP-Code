import { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { fetchDataSummary } from '../api';

const COLORS = {
  'Efficient': '#22c55e',
  'Adequate': '#3b82f6',
  'Underlit': '#ef4444',
  'Overlit': '#f59e0b',
  'No Nearby Light': '#64748b',
};

export default function EfficiencyPanel() {
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    fetchDataSummary().then(setSummary);
  }, []);

  if (!summary) {
    return <div className="p-6 text-slate-400">Loading...</div>;
  }

  const pieData = Object.entries(summary.efficiency_breakdown || {}).map(([name, value]) => ({
    name, value,
  }));

  const total = pieData.reduce((s, d) => s + d.value, 0);

  return (
    <div className="p-6 space-y-8 overflow-y-auto h-full">
      {/* Stats cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Pedestrian Sensors" value={summary.pedestrian_sensors} />
        <StatCard label="Streetlights" value={summary.streetlights?.toLocaleString()} />
        <StatCard label="Avg Lux Level" value={summary.avg_lux} />
        <StatCard label="Records Analyzed" value={summary.pedestrian_records?.toLocaleString()} />
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Pie chart */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
          <h3 className="text-white font-semibold mb-3">Lighting Efficiency Distribution</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={100}
                label={({ name, value }) => `${name} (${Math.round(value / total * 100)}%)`}
                labelLine={{ stroke: '#64748b' }}
              >
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={COLORS[entry.name] || '#64748b'} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: 8 }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Breakdown table */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
          <h3 className="text-white font-semibold mb-3">Classification Breakdown</h3>
          <div className="space-y-3">
            {pieData.map(({ name, value }) => (
              <div key={name}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-300">{name}</span>
                  <span className="text-slate-400">{value} ({Math.round(value / total * 100)}%)</span>
                </div>
                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${(value / total) * 100}%`,
                      backgroundColor: COLORS[name] || '#64748b',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 p-3 bg-slate-700/30 rounded-lg">
            <h4 className="text-sm font-medium text-white mb-2">Methodology</h4>
            <ul className="text-xs text-slate-400 space-y-1">
              <li><span className="text-green-400">Efficient:</span> Traffic 300+/hr with 7+ lux (P3)</li>
              <li><span className="text-blue-400">Adequate:</span> Moderate traffic, sufficient lighting</li>
              <li><span className="text-red-400">Underlit:</span> High traffic but insufficient lux</li>
              <li><span className="text-amber-400">Overlit:</span> Low traffic but exceeds P1 (14 lux)</li>
              <li><span className="text-slate-400">No Nearby Light:</span> Nearest streetlight &gt;200m</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Top sensors */}
      <div>
        <h3 className="text-white font-semibold mb-3">Top 10 Busiest Pedestrian Sensors</h3>
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-700/50">
                <th className="px-4 py-2 text-left text-slate-300">#</th>
                <th className="px-4 py-2 text-left text-slate-300">Sensor</th>
                <th className="px-4 py-2 text-right text-slate-300">Avg/hr</th>
                <th className="px-4 py-2 text-right text-slate-300">Peak</th>
                <th className="px-4 py-2 text-right text-slate-300">Location</th>
              </tr>
            </thead>
            <tbody>
              {summary.top_sensors?.map((s, i) => (
                <tr key={i} className="border-t border-slate-700/50 hover:bg-slate-700/30">
                  <td className="px-4 py-2 text-slate-500">{i + 1}</td>
                  <td className="px-4 py-2 text-white font-medium">{s.name}</td>
                  <td className="px-4 py-2 text-right text-amber-400 font-mono">{s.avg_traffic}</td>
                  <td className="px-4 py-2 text-right text-slate-400 font-mono">{s.max_traffic}</td>
                  <td className="px-4 py-2 text-right text-slate-500 text-xs font-mono">
                    {s.lat.toFixed(4)}, {s.lon.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="text-xs text-slate-400 mt-1">{label}</div>
    </div>
  );
}
