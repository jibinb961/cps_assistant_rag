import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';

interface DisambiguationModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  programs: string[];
  onSelectProgram: (program: string) => void;
}

export const DisambiguationModal: React.FC<DisambiguationModalProps> = ({
  open,
  onOpenChange,
  programs,
  onSelectProgram,
}) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Choose a Program</DialogTitle>
          <DialogDescription>
            Your question matches multiple programs. Please select the specific program you're interested in:
          </DialogDescription>
        </DialogHeader>
        
        <ScrollArea className="h-[300px] mt-4">
          <div className="flex flex-col gap-2">
            {programs.map((program) => (
              <Button
                key={program}
                variant="outline"
                className="justify-start text-left h-auto py-3"
                onClick={() => onSelectProgram(program)}
              >
                {program}
              </Button>
            ))}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}; 