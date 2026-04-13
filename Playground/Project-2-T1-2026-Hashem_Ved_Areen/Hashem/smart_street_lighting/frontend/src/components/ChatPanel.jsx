import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Lightbulb, Download } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { createDesign } from '../api';
import DesignMap from './DesignMap';

const EXAMPLES = [
  "Design lighting for a 200m pathway in Fitzroy Gardens with high evening traffic",
  "Plan lighting for a quiet 150m path in Royal Park",
  "Recommend lighting for a 300m busy shared path near Melbourne CBD",
  "Design energy-efficient lighting for a 250m pathway in Princes Park",
];

export default function ChatPanel({ messages, setMessages, conversationId, onConversationUpdate }) {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [lastDesign, setLastDesign] = useState(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function handleSubmit(query) {
    const q = query || input.trim();
    if (!q) return;

    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setInput('');
    setLoading(true);

    try {
      const data = await createDesign(q, true, conversationId);
      setLastDesign(data);

      // Update conversation ID from response (auto-created on first message)
      if (data.conversation_id && data.conversation_id !== conversationId) {
        onConversationUpdate(data.conversation_id);
      }

      const report = data.llm_report || data.calculation_report;
      const sources = data.sources?.length
        ? '\n\n---\n**Sources:** ' + data.sources.map((s) => `${s.source} (${s.score})`).join(', ')
        : '';
      const summary = `\n\n---\n**Quick Summary:** ${data.design.num_lights} x ${data.design.led_wattage}W LED | Spacing: ${data.design.spacing_m}m | Category: ${data.design.p_category} | Annual cost: $${data.design.annual_energy_cost_aud} | Saving vs HPS: ${data.design.energy_saving_vs_hps_percent}%`;

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: report + sources + summary,
        design: data.design,
        mapData: data.map_data || null,
        safetyContext: data.safety_context || null,
        spatialContext: data.spatial_context || null,
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.message}. Make sure the API server is running (uvicorn api:app --port 8000).`,
      }]);
    } finally {
      setLoading(false);
    }
  }

  function handleDownload() {
    if (!lastDesign) return;
    const text = `SMART STREET LIGHTING DESIGN REPORT\n${'='.repeat(50)}\n\n${lastDesign.calculation_report}\n\n${lastDesign.llm_report || ''}`;
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `design_report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 bg-amber-500/10 rounded-2xl flex items-center justify-center mb-4">
              <Lightbulb className="w-8 h-8 text-amber-400" />
            </div>
            <h2 className="text-xl font-semibold text-white mb-2">Street Lighting Design Assistant</h2>
            <p className="text-slate-400 text-sm max-w-md mb-6">
              Describe your location and requirements in natural language.
              The system uses real Melbourne data and AS/NZS 1158 standards.
            </p>
            <div className="grid grid-cols-2 gap-2 max-w-lg">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => handleSubmit(ex)}
                  className="text-left text-xs p-3 rounded-lg bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700 hover:text-white transition-colors"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-xl px-4 py-3 text-sm ${
              msg.role === 'user'
                ? 'bg-amber-500 text-slate-900'
                : 'bg-slate-800 border border-slate-700 text-slate-200'
            }`}>
              {msg.role === 'assistant' ? (
                <div className="prose prose-sm prose-invert max-w-none prose-p:my-1 prose-li:my-0 prose-headings:text-amber-400 prose-strong:text-white prose-hr:border-slate-600">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              ) : (
                <p>{msg.content}</p>
              )}
              {msg.design && (
                <div className="mt-3 pt-3 border-t border-slate-600 grid grid-cols-3 gap-2">
                  <Stat label="Lights" value={msg.design.num_lights} />
                  <Stat label="Spacing" value={`${msg.design.spacing_m}m`} />
                  <Stat label="Category" value={msg.design.p_category} />
                  <Stat label="Annual Cost" value={`$${msg.design.annual_energy_cost_aud}`} />
                  <Stat label="HPS Saving" value={`${msg.design.energy_saving_vs_hps_percent}%`} />
                  <Stat label="CO2 Saved" value={`${msg.design.co2_saving_vs_hps_kg}kg`} />
                </div>
              )}
              {msg.mapData && (
                <DesignMap
                  pathwayGeojson={msg.mapData.pathway_geojson}
                  lightPositions={msg.design?.light_positions}
                  parkBoundary={msg.mapData.park_boundary}
                  existingLights={msg.spatialContext?.existing_lights_on_path}
                />
              )}
              {msg.safetyContext && msg.safetyContext.risk_category !== 'low' && (
                <div className="mt-2 px-2 py-1.5 rounded bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300">
                  Safety: {msg.safetyContext.risk_category} risk ({msg.safetyContext.lga_name}) — P-category adjusted by {msg.safetyContext.p_category_adjustment}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 flex items-center gap-2 text-sm text-slate-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              Analyzing location and generating design...
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-slate-700">
        {lastDesign && (
          <button
            onClick={handleDownload}
            className="mb-2 flex items-center gap-1 text-xs text-amber-400 hover:text-amber-300"
          >
            <Download className="w-3 h-3" /> Download last report
          </button>
        )}
        <form
          onSubmit={e => { e.preventDefault(); handleSubmit(); }}
          className="flex gap-2"
        >
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Design lighting for a 200m pathway in Fitzroy Gardens..."
            className="flex-1 bg-slate-800 border border-slate-600 rounded-lg px-4 py-2.5 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:border-amber-500"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-amber-500 hover:bg-amber-400 disabled:bg-slate-700 disabled:text-slate-500 text-slate-900 px-4 py-2.5 rounded-lg transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="text-center">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="text-sm font-semibold text-amber-400">{value}</div>
    </div>
  );
}
