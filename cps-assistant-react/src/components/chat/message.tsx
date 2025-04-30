import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Card } from '@/components/ui/card';
import { Message as MessageType } from '../../types';
import { format } from 'date-fns';

interface MessageProps {
  message: MessageType;
}

export const Message: React.FC<MessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`flex ${isUser ? 'flex-row-reverse' : 'flex-row'} gap-3 max-w-[80%]`}>
        <Avatar className="h-8 w-8">
          {isUser ? (
            <>
              <AvatarFallback>U</AvatarFallback>
              <AvatarImage src="/avatars/user.png" alt="User" />
            </>
          ) : (
            <>
              <AvatarFallback>AI</AvatarFallback>
              <AvatarImage src="/avatars/assistant.png" alt="Assistant" />
            </>
          )}
        </Avatar>
        
        <div>
          <Card className={`p-3 ${isUser ? 'bg-primary text-primary-foreground' : 'bg-muted'}`}>
            <div className="prose dark:prose-invert prose-sm break-words">
              <ReactMarkdown>
                {message.content}
              </ReactMarkdown>
            </div>
          </Card>
          <p className="text-xs text-muted-foreground mt-1">
            {format(new Date(message.createdAt), 'h:mm a')}
          </p>
        </div>
      </div>
    </div>
  );
}; 