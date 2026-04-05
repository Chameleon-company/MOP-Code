import { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';
import TrafficPanel from './components/TrafficPanel';
import EfficiencyPanel from './components/EfficiencyPanel';
import DataPanel from './components/DataPanel';
import { fetchConversations, fetchConversation, deleteConversation } from './api';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);

  const loadConversations = useCallback(async () => {
    try {
      const convs = await fetchConversations();
      setConversations(convs);
    } catch (err) {
      console.error('Failed to load conversations:', err);
    }
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  async function handleSelectConversation(id) {
    try {
      const conv = await fetchConversation(id);
      setActiveConversationId(id);
      setChatMessages(
        conv.messages.map(m => ({
          role: m.role,
          content: m.content,
          design: m.design_data?.design || null,
        }))
      );
      setActiveTab('chat');
    } catch (err) {
      console.error('Failed to load conversation:', err);
    }
  }

  function handleNewConversation() {
    setActiveConversationId(null);
    setChatMessages([]);
    setActiveTab('chat');
  }

  async function handleDeleteConversation(id) {
    try {
      await deleteConversation(id);
      if (activeConversationId === id) {
        setActiveConversationId(null);
        setChatMessages([]);
      }
      loadConversations();
    } catch (err) {
      console.error('Failed to delete conversation:', err);
    }
  }

  function handleConversationUpdate(convId) {
    setActiveConversationId(convId);
    loadConversations();
  }

  const panels = {
    chat: (
      <ChatPanel
        messages={chatMessages}
        setMessages={setChatMessages}
        conversationId={activeConversationId}
        onConversationUpdate={handleConversationUpdate}
      />
    ),
    traffic: <TrafficPanel />,
    efficiency: <EfficiencyPanel />,
    data: <DataPanel />,
  };

  return (
    <div className="flex h-screen bg-slate-950">
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
      />
      <main className="flex-1 overflow-hidden">
        {panels[activeTab]}
      </main>
    </div>
  );
}

export default App;
