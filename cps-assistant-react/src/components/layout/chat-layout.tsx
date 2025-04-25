import React, { useEffect } from 'react';
import { SessionList } from '../sidebar/session-list';
import { ChatContainer } from '../chat/chat-container';
import { useChatStore } from '../../lib/store/chat-store';
import { chatApi } from '../../lib/api';
import { v4 as uuidv4 } from 'uuid';

export const ChatLayout: React.FC = () => {
  const sessions = useChatStore((state) => state.sessions);
  const activeSessionId = useChatStore((state) => state.activeSessionId);
  const messages = useChatStore((state) => state.messages);
  const isLoading = useChatStore((state) => state.isLoading);
  const programContext = useChatStore((state) => state.programContext);
  
  const setSessions = useChatStore((state) => state.setSessions);
  const setActiveSessionId = useChatStore((state) => state.setActiveSessionId);
  const setMessages = useChatStore((state) => state.setMessages);
  const addSession = useChatStore((state) => state.addSession);
  
  // Fetch sessions on initial load
  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const sessions = await chatApi.getSessions();
        setSessions(sessions);
        
        // If there are sessions, set the first one as active
        if (sessions.length > 0 && !activeSessionId) {
          setActiveSessionId(sessions[0].id);
        }
      } catch (error) {
        console.error('Error fetching sessions:', error);
      }
    };
    
    fetchSessions();
  }, []);
  
  // Fetch messages when active session changes
  useEffect(() => {
    if (!activeSessionId) return;
    
    const fetchMessages = async () => {
      try {
        const messages = await chatApi.getMessages(activeSessionId);
        setMessages(activeSessionId, messages);
      } catch (error) {
        console.error('Error fetching messages:', error);
      }
    };
    
    fetchMessages();
  }, [activeSessionId]);
  
  const handleNewSession = async () => {
    try {
      // Create a new temporary ID for optimistic UI
      const tempId = uuidv4();
      const newSession = {
        id: tempId,
        title: 'New Conversation',
        createdAt: new Date().toISOString(),
        lastActive: new Date().toISOString()
      };
      
      // Add session to UI immediately
      addSession(newSession);
      setActiveSessionId(newSession.id);
      
      // Create session on the server
      const createdSession = await chatApi.createSession();
      
      // Update UI with the actual session ID
      setSessions(
        sessions.map(session => 
          session.id === tempId ? createdSession : session
        )
      );
      setActiveSessionId(createdSession.id);
    } catch (error) {
      console.error('Error creating session:', error);
    }
  };
  
  const handleSelectSession = (sessionId: string) => {
    setActiveSessionId(sessionId);
  };
  
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar - fixed width, full height */}
      <div className="w-64 h-full border-r bg-muted/40 dark:bg-muted/20 flex-shrink-0 overflow-y-auto">
        <SessionList
          sessions={sessions}
          activeSessionId={activeSessionId}
          onNewSession={handleNewSession}
          onSelectSession={handleSelectSession}
        />
      </div>
      
      {/* Main content - dynamic width, fixed height */}
      <div className="flex-1 h-full overflow-hidden">
        {activeSessionId ? (
          <ChatContainer
            sessionId={activeSessionId}
            messages={messages[activeSessionId] || []}
            isLoading={isLoading}
            programContext={programContext}
          />
        ) : (
          <div className="flex h-full items-center justify-center p-4 text-center">
            <div>
              <h2 className="text-2xl font-bold mb-2">Welcome to CPS AI Assistant</h2>
              <p className="text-muted-foreground mb-4">
                Start a new conversation or select an existing one from the sidebar.
              </p>
              <button
                className="px-4 py-2 bg-primary text-primary-foreground rounded-md"
                onClick={handleNewSession}
              >
                Start New Chat
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 