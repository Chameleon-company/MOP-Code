import { Lightbulb, MessageSquare, BarChart3, PieChart, Database, Plus, Trash2 } from 'lucide-react';

const navItems = [
  { id: 'chat', label: 'Design Assistant', icon: MessageSquare },
  { id: 'traffic', label: 'Traffic Analysis', icon: BarChart3 },
  { id: 'efficiency', label: 'Efficiency', icon: PieChart },
  { id: 'data', label: 'Data Explorer', icon: Database },
];

export default function Sidebar({
  activeTab,
  onTabChange,
  conversations = [],
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
}) {
  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-700 flex flex-col">
      <div className="p-5 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-amber-500 rounded-lg flex items-center justify-center">
            <Lightbulb className="w-6 h-6 text-slate-900" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white leading-tight">Smart Street</h1>
            <h1 className="text-sm font-bold text-amber-400 leading-tight">Lighting</h1>
          </div>
        </div>
        <p className="text-[11px] text-slate-400 mt-2">AI-powered design for Melbourne</p>
      </div>

      <nav className="p-3 space-y-1">
        {navItems.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onTabChange(id)}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
              activeTab === id
                ? 'bg-amber-500/15 text-amber-400 font-medium'
                : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
            }`}
          >
            <Icon className="w-4 h-4 flex-shrink-0" />
            {label}
          </button>
        ))}
      </nav>

      {/* Conversations */}
      <div className="flex-1 overflow-hidden flex flex-col border-t border-slate-700">
        <div className="px-3 pt-3 pb-1 flex items-center justify-between">
          <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider">History</span>
          <button
            onClick={onNewConversation}
            className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-amber-400 transition-colors"
            title="New conversation"
          >
            <Plus className="w-3.5 h-3.5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
          {conversations.length === 0 ? (
            <p className="text-[11px] text-slate-600 px-2 py-3">No conversations yet</p>
          ) : (
            conversations.map(conv => (
              <div
                key={conv.id}
                className={`group flex items-center gap-1 rounded-lg transition-colors ${
                  activeConversationId === conv.id
                    ? 'bg-slate-800 text-white'
                    : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-300'
                }`}
              >
                <button
                  onClick={() => onSelectConversation(conv.id)}
                  className="flex-1 text-left px-2.5 py-2 text-xs truncate"
                  title={conv.title}
                >
                  {conv.title}
                </button>
                <button
                  onClick={e => {
                    e.stopPropagation();
                    onDeleteConversation(conv.id);
                  }}
                  className="p-1.5 mr-1 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/20 hover:text-red-400 transition-all"
                  title="Delete"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="p-4 border-t border-slate-700">
        <div className="text-[10px] text-slate-500 space-y-1">
          <p>AS/NZS 1158 Standards</p>
          <p>Melbourne Open Data</p>
          <p>Local LLM via LM Studio</p>
        </div>
      </div>
    </aside>
  );
}
