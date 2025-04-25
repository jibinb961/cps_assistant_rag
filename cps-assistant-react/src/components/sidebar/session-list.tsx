import React from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { ChatSession } from '../../types';
import { FiPlus, FiMessageSquare } from 'react-icons/fi';
import { format } from 'date-fns';

interface SessionListProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
}

export const SessionList: React.FC<SessionListProps> = ({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession
}) => {
  return (
    <div className="h-full flex flex-col">
      <div className="p-4">
        <Button 
          onClick={onNewSession} 
          className="w-full"
          size="sm"
          variant="default"
        >
          <FiPlus className="mr-2" /> New Chat
        </Button>
      </div>
      <Separator />
      
      <ScrollArea className="flex-1">
        {sessions.length > 0 ? (
          <div className="p-2">
            {sessions.map((session) => (
              <Button
                key={session.id}
                variant={activeSessionId === session.id ? "secondary" : "ghost"}
                className="w-full justify-start mb-1 text-left overflow-hidden"
                onClick={() => onSelectSession(session.id)}
              >
                <div className="flex items-center truncate w-full">
                  <FiMessageSquare className="mr-2 flex-shrink-0" />
                  <div className="truncate">
                    <span className="truncate block">{session.title}</span>
                    <span className="text-xs text-muted-foreground truncate block">
                      {format(new Date(session.lastActive), 'MMM d, h:mm a')}
                    </span>
                  </div>
                </div>
              </Button>
            ))}
          </div>
        ) : (
          <div className="p-4 text-center text-muted-foreground">
            No conversations yet
          </div>
        )}
      </ScrollArea>
    </div>
  );
}; 