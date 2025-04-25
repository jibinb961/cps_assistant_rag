import React, { useEffect, useRef } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Message } from './message';
import { ChatInput } from './chat-input';
import { DisambiguationModal } from './disambiguation-modal';
import { Message as MessageType } from '../../types';
import { useChatStore } from '../../lib/store/chat-store';
import { chatApi } from '../../lib/api';
import { v4 as uuidv4 } from 'uuid';

interface ChatContainerProps {
  sessionId: string;
  messages: MessageType[];
  isLoading: boolean;
  programContext?: string | null;
}

export const ChatContainer: React.FC<ChatContainerProps> = ({
  sessionId,
  messages,
  isLoading,
  programContext
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const showDisambiguation = useChatStore((state) => state.showDisambiguation);
  const programOptions = useChatStore((state) => state.programOptions);
  const pendingMessage = useChatStore((state) => state.pendingMessage);
  const setShowDisambiguation = useChatStore((state) => state.setShowDisambiguation);
  const setProgramContext = useChatStore((state) => state.setProgramContext);
  const setPendingMessage = useChatStore((state) => state.setPendingMessage);
  const setLoading = useChatStore((state) => state.setLoading);
  const addMessage = useChatStore((state) => state.addMessage);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    if (!sessionId) return;
    
    // Add user message to UI immediately
    const userMessage: MessageType = {
      id: uuidv4(),
      role: 'user',
      content,
      createdAt: new Date().toISOString()
    };
    
    addMessage(sessionId, userMessage);
    setLoading(true);
    
    try {
      // Send message to API
      const response = await chatApi.sendMessage({
        sessionId,
        message: content
      });
      
      // Add AI response to UI
      addMessage(sessionId, response);
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message
      addMessage(sessionId, {
        id: uuidv4(),
        role: 'assistant',
        content: 'Sorry, an error occurred while processing your message. Please try again.',
        createdAt: new Date().toISOString()
      });
    } finally {
      setLoading(false);
    }
  };
  
  const handleProgramSelection = async (program: string) => {
    // Set program context
    setProgramContext(program);
    setShowDisambiguation(false);
    
    // If there's a pending message, send it with the newly selected program
    if (pendingMessage && sessionId) {
      await chatApi.setSessionContext({
        sessionId,
        programName: program
      });
      
      handleSendMessage(pendingMessage);
      setPendingMessage(null);
    }
  };
  
  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Program context indicator - fixed at top */}
      {programContext && (
        <div className="bg-muted text-muted-foreground text-sm py-1 px-4 border-b flex-shrink-0">
          Currently viewing: <span className="font-semibold">{programContext}</span>
        </div>
      )}
      
      {/* Chat messages - scrollable area */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-4">
            <h2 className="text-2xl font-bold mb-2">CPS AI Assistant</h2>
            <p className="text-muted-foreground mb-4 max-w-md">
              Ask me anything about CPS programs, courses, requirements, or co-op opportunities.
            </p>
          </div>
        ) : (
          <div>
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="flex items-center gap-2 text-muted-foreground animate-pulse">
                <div className="h-2 w-2 bg-current rounded-full"/>
                <div className="h-2 w-2 bg-current rounded-full delay-75"/>
                <div className="h-2 w-2 bg-current rounded-full delay-150"/>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
      
      {/* Chat input - fixed at bottom */}
      <div className="border-t flex-shrink-0">
        <ChatInput 
          onSendMessage={handleSendMessage} 
          disabled={isLoading || showDisambiguation}
        />
      </div>
      
      {/* Disambiguation modal */}
      <DisambiguationModal
        open={showDisambiguation}
        onOpenChange={setShowDisambiguation}
        programs={programOptions}
        onSelectProgram={handleProgramSelection}
      />
    </div>
  );
}; 