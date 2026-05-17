import { useState, useEffect } from 'react';
import { fetchCategories, fetchDataSummary } from '../api';

export default function DataPanel() {
  const [categories, setCategories] = useState(null);
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    fetchCategories().then(setCategories);
    fetchDataSummary().then(setSummary);
  }, []);

  return (
    <div className="p-6 space-y-8 overflow-y-auto h-full">
      {/* AS/NZS 1158 Categories */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-1">AS/NZS 1158 Lighting Categories</h2>
        <p className="text-sm text-slate-400 mb-4">
          Reference table of P-category requirements used by the calculation engine.
        </p>
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-700/50">
                <th className="px-4 py-2 text-left text-slate-300">Category</th>
                <th className="px-4 py-2 text-left text-slate-300">Application</th>
                <th className="px-4 py-2 text-right text-slate-300">Avg Lux</th>
                <th className="px-4 py-2 text-right text-slate-300">Min Lux</th>
                <th className="px-4 py-2 text-right text-slate-300">Uniformity</th>
              </tr>
            </thead>
            <tbody>
              {categories && Object.entries(categories).map(([id, cat]) => (
                <tr key={id} className="border-t border-slate-700/50 hover:bg-slate-700/30">
                  <td className="px-4 py-2 text-amber-400 font-mono font-bold">{id}</td>
                  <td className="px-4 py-2 text-slate-300">{cat.name}</td>
                  <td className="px-4 py-2 text-right text-white font-mono">{cat.avg_lux}</td>
                  <td className="px-4 py-2 text-right text-slate-400 font-mono">{cat.min_lux}</td>
                  <td className="px-4 py-2 text-right text-slate-400 font-mono">{cat.uniformity}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* System Architecture */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-3">System Architecture</h2>
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'Data Layer', desc: 'Melbourne Open Data API — pedestrian counts + streetlight positions', color: 'blue' },
            { label: 'Spatial Analysis', desc: 'k-NN matching (haversine), efficiency classification', color: 'green' },
            { label: 'Calculation Engine', desc: 'AS/NZS 1158 standards — deterministic light placement + energy cost', color: 'amber' },
            { label: 'RAG + LLM', desc: 'ChromaDB vector store + Llama 3.1 8B — explained design reports', color: 'purple' },
          ].map(({ label, desc, color }) => (
            <div key={label} className={`bg-slate-800/50 border border-slate-700 rounded-xl p-4`}>
              <div className={`text-sm font-semibold mb-1 ${
                color === 'blue' ? 'text-blue-400' :
                color === 'green' ? 'text-green-400' :
                color === 'amber' ? 'text-amber-400' : 'text-purple-400'
              }`}>{label}</div>
              <p className="text-xs text-slate-400">{desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Key findings */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-3">Key Findings from Data</h2>
        <div className="grid grid-cols-3 gap-4">
          <FindingCard
            title="8.1% Underlit"
            desc="High-traffic locations with insufficient lighting below AS/NZS 1158 P-category requirements"
            severity="critical"
          />
          <FindingCard
            title="57-66% Energy Saving"
            desc="LED replacement of HPS luminaires reduces energy consumption by more than half"
            severity="positive"
          />
          <FindingCard
            title="36% Dimming Saving"
            desc="Additional energy saving from adaptive dimming based on real hourly traffic patterns"
            severity="positive"
          />
          <FindingCard
            title="9.6yr Retrofit Payback"
            desc="LED retrofit payback period based on energy + maintenance savings (luminaire swap)"
            severity="neutral"
          />
          <FindingCard
            title="Solar Marginal"
            desc="Melbourne's winter solar (1.6 peak hours/day) makes standalone solar LED impractical"
            severity="warning"
          />
          <FindingCard
            title="3000K Recommended"
            desc="Warm white colour temperature per City of Melbourne ecological guidelines"
            severity="neutral"
          />
        </div>
      </div>
    </div>
  );
}

function FindingCard({ title, desc, severity }) {
  const borderColor = {
    critical: 'border-red-500/50',
    positive: 'border-green-500/50',
    warning: 'border-amber-500/50',
    neutral: 'border-slate-600',
  }[severity];

  const titleColor = {
    critical: 'text-red-400',
    positive: 'text-green-400',
    warning: 'text-amber-400',
    neutral: 'text-slate-300',
  }[severity];

  return (
    <div className={`bg-slate-800/50 border ${borderColor} rounded-xl p-4`}>
      <div className={`text-sm font-bold ${titleColor}`}>{title}</div>
      <p className="text-xs text-slate-400 mt-1">{desc}</p>
    </div>
  );
}
