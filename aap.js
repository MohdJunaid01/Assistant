import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import ChatInterface from './components/ChatInterface';
import TradingIdeas from './components/TradingIdeas';
import VoiceControls from './components/VoiceControls';
import VisualControls from './components/VisualControls';
import StatusBar from './components/StatusBar';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [assistantStatus, setAssistantStatus] = useState('offline');
  const [conversations, setConversations] = useState([]);
  const [tradingIdeas, setTradingIdeas] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    checkAssistantStatus();
    loadTradingIdeas();
    const interval = setInterval(checkAssistantStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkAssistantStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/public`);
      if (response.ok) {
        setAssistantStatus('online');
        const data = await response.json();
        setTradingIdeas(data.recent_ideas || []);
      } else {
        setAssistantStatus('offline');
      }
    } catch (error) {
      setAssistantStatus('offline');
    }
  };

  const loadTradingIdeas = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/public`);
      if (response.ok) {
        const data = await response.json();
        setTradingIdeas(data.recent_ideas || []);
      }
    } catch (error) {
      console.error('Failed to load trading ideas:', error);
    }
  };

  const sendMessage = async (message, type = 'text') => {
    setIsLoading(true);
    const newMessage = {
      id: Date.now(),
      type: 'user',
      content: message,
      inputType: type,
      timestamp: new Date().toLocaleTimeString()
    };

    setConversations(prev => [...prev, newMessage]);

    try {
      // Simulate assistant response (replace with actual API call)
      const response = await simulateAssistantResponse(message, type);
      
      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: response,
        timestamp: new Date().toLocaleTimeString()
      };

      setConversations(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: 'Sorry, I encountered an error processing your request.',
        timestamp: new Date().toLocaleTimeString()
      };
      setConversations(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const simulateAssistantResponse = async (message, type) => {
    // This would be replaced with actual API calls to your Python backend
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const responses = {
      text: {
        'hello': 'Hey! What\'s on your mind today? A project, trading, or something else?',
        'trading': 'Let\'s dive into trading! Want to brainstorm a new strategy or tweak your bot?',
        'help': 'I can help with projects, trading ideas, voice commands, and visual gestures. What would you like to explore?',
        'default': 'I\'m here to help! What would you like to discuss?'
      },
      audio: 'I heard you loud and clear! How can I assist you today?',
      visual: 'I see you! What gesture would you like me to recognize?'
    };

    const lowerMessage = message.toLowerCase();
    if (type === 'text') {
      for (const [key, response] of Object.entries(responses.text)) {
        if (lowerMessage.includes(key)) {
          return response;
        }
      }
      return responses.text.default;
    }
    
    return responses[type] || responses.text.default;
  };

  const addTradingIdea = async (idea) => {
    try {
      const response = await fetch(`${API_BASE_URL}/public/trading`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ idea }),
      });

      if (response.ok) {
        loadTradingIdeas();
        return true;
      }
    } catch (error) {
      console.error('Failed to add trading idea:', error);
    }
    return false;
  };

  const provideFeedback = (messageId, feedback) => {
    // Implement feedback functionality
    console.log(`Feedback for message ${messageId}: ${feedback}`);
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🤖 Intelligent Assistant</h1>
        <StatusBar status={assistantStatus} />
      </header>

      <nav className="tab-navigation">
        <button 
          className={activeTab === 'chat' ? 'active' : ''}
          onClick={() => setActiveTab('chat')}
        >
          💬 Chat
        </button>
        <button 
          className={activeTab === 'trading' ? 'active' : ''}
          onClick={() => setActiveTab('trading')}
        >
          📈 Trading Ideas
        </button>
        <button 
          className={activeTab === 'voice' ? 'active' : ''}
          onClick={() => setActiveTab('voice')}
        >
          🎤 Voice
        </button>
        <button 
          className={activeTab === 'visual' ? 'active' : ''}
          onClick={() => setActiveTab('visual')}
        >
          👁️ Visual
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'chat' && (
          <ChatInterface 
            conversations={conversations}
            onSendMessage={sendMessage}
            onProvideFeedback={provideFeedback}
            isLoading={isLoading}
          />
        )}
        {activeTab === 'trading' && (
          <TradingIdeas 
            ideas={tradingIdeas}
            onAddIdea={addTradingIdea}
          />
        )}
        {activeTab === 'voice' && (
          <VoiceControls 
            onSendMessage={sendMessage}
            isLoading={isLoading}
          />
        )}
        {activeTab === 'visual' && (
          <VisualControls 
            onSendMessage={sendMessage}
            isLoading={isLoading}
          />
        )}
      </main>
    </div>
  );
}

export default App;
