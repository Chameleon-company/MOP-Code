import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, Cell } from 'recharts';
import { fetchHourlyTraffic, fetchDimmingSchedule, fetchWeekdayWeekend } from '../api';

const OUTPUT_COLORS = { 100: '#22c55e', 80: '#f59e0b', 60: '#f97316', 40: '#ef4444' };

export default function TrafficPanel() {
  const [hourly, setHourly] = useState([]);
  const [schedule, setSchedule] = useState([]);
  const [weekdayWeekend, setWeekdayWeekend] = useState(null);

  useEffect(() => {
    fetchHourlyTraffic().then(d => setHourly(d.profile || []));
    fetchDimmingSchedule().then(d => setSchedule(d.schedule || []));
    fetchWeekdayWeekend().then(d => setWeekdayWeekend(d));
  }, []);

  const chartData = hourly.map(h => {
    const sched = schedule.find(s => s.hour === h.hour);
    return {
      ...h,
      hour_label: `${String(h.hour).padStart(2, '0')}:00`,
      output_pct: sched?.suggested_output_pct || 100,
      fill: OUTPUT_COLORS[sched?.suggested_output_pct] || '#22c55e',
    };
  });

  const wwData = [];
  if (weekdayWeekend) {
    const allHours = new Set();
    Object.values(weekdayWeekend).forEach(arr => arr.forEach(d => allHours.add(d.hour)));
    [...allHours].sort((a, b) => a - b).forEach(hour => {
      const entry = { hour: `${String(hour).padStart(2, '0')}:00` };
      Object.entries(weekdayWeekend).forEach(([dayType, arr]) => {
        const match = arr.find(d => d.hour === hour);
        entry[dayType] = match?.avg_traffic || 0;
      });
      wwData.push(entry);
    });
  }

  return (
    <div className="p-6 space-y-8 overflow-y-auto h-full">
      <div>
        <h2 className="text-lg font-semibold text-white mb-1">Hourly Pedestrian Traffic</h2>
        <p className="text-sm text-slate-400 mb-4">
          Bar color indicates suggested lighting output level based on AS/NZS 1158 dimming provisions.
        </p>
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="hour_label" tick={{ fill: '#94a3b8', fontSize: 11 }} interval={1} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: 8 }}
                labelStyle={{ color: '#f8fafc' }}
                itemStyle={{ color: '#fbbf24' }}
                formatter={(val, name) => [Math.round(val), name === 'avg_traffic' ? 'Avg Traffic' : name]}
              />
              <Bar dataKey="avg_traffic" name="Avg Traffic" radius={[4, 4, 0, 0]}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-3 justify-center">
            {Object.entries(OUTPUT_COLORS).map(([pct, color]) => (
              <div key={pct} className="flex items-center gap-1.5 text-xs text-slate-400">
                <div className="w-3 h-3 rounded" style={{ backgroundColor: color }} />
                {pct}% output
              </div>
            ))}
          </div>
        </div>
      </div>

      {wwData.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-white mb-1">Weekday vs Weekend Patterns</h2>
          <p className="text-sm text-slate-400 mb-4">
            Different traffic patterns inform time-of-week dimming schedules.
          </p>
          <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={wwData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="hour" tick={{ fill: '#94a3b8', fontSize: 11 }} interval={2} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: 8 }} />
                <Legend />
                <Line type="monotone" dataKey="Weekday" stroke="#3b82f6" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Weekend" stroke="#f59e0b" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div>
        <h2 className="text-lg font-semibold text-white mb-3">Dimming Schedule</h2>
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-700/50">
                <th className="px-4 py-2 text-left text-slate-300">Hour</th>
                <th className="px-4 py-2 text-right text-slate-300">Avg Traffic</th>
                <th className="px-4 py-2 text-right text-slate-300">Output</th>
                <th className="px-4 py-2 text-left text-slate-300">Reason</th>
              </tr>
            </thead>
            <tbody>
              {schedule.map((row, i) => (
                <tr key={i} className="border-t border-slate-700/50 hover:bg-slate-700/30">
                  <td className="px-4 py-1.5 text-slate-300 font-mono">{String(row.hour).padStart(2, '0')}:00</td>
                  <td className="px-4 py-1.5 text-right text-slate-400">{row.avg_traffic}</td>
                  <td className="px-4 py-1.5 text-right">
                    <span className="inline-block px-2 py-0.5 rounded text-xs font-medium" style={{
                      backgroundColor: OUTPUT_COLORS[row.suggested_output_pct] + '20',
                      color: OUTPUT_COLORS[row.suggested_output_pct],
                    }}>
                      {row.suggested_output_pct}%
                    </span>
                  </td>
                  <td className="px-4 py-1.5 text-slate-400 text-xs">{row.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
