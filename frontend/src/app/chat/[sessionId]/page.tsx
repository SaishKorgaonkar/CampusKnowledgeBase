'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import ChatSidebar from '@/components/chat/ChatSidebar';
import ChatWindow from '@/components/chat/ChatWindow';
import PDFViewerPanel from '@/components/chat/PDFViewerPanel';
import { Session, Message, Source } from '@/types/chat';
import { mockSessions } from '@/lib/mockData';
import { useSemester } from '@/contexts/SemesterContext';

export default function ChatSessionPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;
  const { course, semesterNumber } = useSemester();

  const [sessions, setSessions] = useState<Session[]>(mockSessions);
  const [messages, setMessages] = useState<{ [key: string]: Message[] }>({});
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [selectedSource, setSelectedSource] = useState<Source | null>(null);
  const [chatWidth, setChatWidth] = useState(50); // Percentage

  // Get current session
  const currentSession = sessions.find((s) => s.id === sessionId);

  // Redirect if session doesn't exist
  useEffect(() => {
    if (!currentSession && sessions.length > 0) {
      router.push(`/chat/${sessions[0].id}`);
    }
  }, [currentSession, sessions, router]);

  const handleNewSession = (subject: string) => {
    const newSession: Session = {
      id: (sessions.length + 1).toString(),
      subject,
      messageCount: 0,
      createdAt: new Date()
    };
    setSessions([...sessions, newSession]);
    router.push(`/chat/${newSession.id}`);
  };

  const handleSendMessage = (content: string) => {
    if (!currentSession) return;

    const userMessage: Message = {
      id: `${Date.now()}-user`,
      role: 'user',
      content,
      timestamp: new Date()
    };

    // TODO: Real API call would be:
    // fetch('/api/ask', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ question: content, course, semester: semesterNumber })
    // })

    // Simulate AI response with sources
    const aiMessage: Message = {
      id: `${Date.now()}-ai`,
      role: 'assistant',
      content: `Thank you for your question about "${content.substring(0, 50)}..."\n\nThis is a mock response. In the production version, this would be powered by RAG technology using your verified campus data and syllabus.\n\nThe system would:\n1. Retrieve relevant information from the knowledge base\n2. Generate accurate, syllabus-aligned answers\n3. Provide exam-focused explanations\n\nYour actual implementation will connect to the Go backend with RAG capabilities.`,
      timestamp: new Date(Date.now() + 1000),
      sources: [
        {
          id: '1',
          fileName: 'DSA_Module_1.pdf',
          title: 'Data Structures - Module 1: Arrays and Linked Lists',
          pageNumber: 27,
          relevance: 0.95,
          excerpt: 'An array is a collection of elements identified by index or key. Arrays store data elements based on sequential, most commonly 0 based, indexes.'
        },
        {
          id: '2',
          fileName: 'DSA_Lecture_Notes_2024.pdf',
          title: 'DSA Complete Lecture Notes - Prof. Kumar',
          pageNumber: 142,
          relevance: 0.87,
          excerpt: 'Time complexity analysis is crucial for understanding algorithm efficiency. Big O notation provides an upper bound on the growth rate of an algorithm.'
        },
        {
          id: '3',
          fileName: 'Previous_Year_Questions.pdf',
          title: 'DSA Previous Year Question Papers (2020-2024)',
          pageNumber: 8,
          relevance: 0.78,
          excerpt: 'This question appeared in the December 2023 examination. Students should focus on understanding the fundamental concepts rather than memorizing solutions.'
        }
      ]
    };

    const sessionMessages = messages[sessionId] || [];
    setMessages({
      ...messages,
      [sessionId]: [...sessionMessages, userMessage, aiMessage]
    });

    // Update message count
    setSessions(
      sessions.map((s) =>
        s.id === sessionId
          ? { ...s, messageCount: s.messageCount + 2 }
          : s
      )
    );
  };

  if (!currentSession) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#f8f9fa]">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-[#1a73e8] border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading session...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <ChatSidebar 
        sessions={sessions} 
        onNewSession={handleNewSession}
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <div className="flex-1 flex overflow-hidden">
        <div style={{ width: selectedSource ? `${chatWidth}%` : '100%' }}>
          <ChatWindow
            subjectName={currentSession.subject}
            messages={messages[sessionId] || []}
            onSendMessage={handleSendMessage}
            onSourceClick={setSelectedSource}
          />
        </div>
        {selectedSource && (
          <>
            {/* Resizable Divider */}
            <div 
              className="w-1 bg-gray-200 hover:bg-[#1a73e8] cursor-col-resize transition-colors flex-shrink-0"
              onMouseDown={(e) => {
                e.preventDefault();
                const startX = e.clientX;
                const startWidth = chatWidth;
                
                const handleMouseMove = (e: MouseEvent) => {
                  requestAnimationFrame(() => {
                    const containerWidth = window.innerWidth - (sidebarCollapsed ? 64 : 320);
                    const delta = ((e.clientX - startX) / containerWidth) * 100;
                    const newWidth = Math.max(30, Math.min(70, startWidth + delta));
                    setChatWidth(newWidth);
                  });
                };
                
                const handleMouseUp = () => {
                  document.removeEventListener('mousemove', handleMouseMove);
                  document.removeEventListener('mouseup', handleMouseUp);
                  document.body.style.cursor = '';
                  document.body.style.userSelect = '';
                };
                
                document.body.style.cursor = 'col-resize';
                document.body.style.userSelect = 'none';
                document.addEventListener('mousemove', handleMouseMove);
                document.addEventListener('mouseup', handleMouseUp);
              }}
            />
            <div style={{ width: `${100 - chatWidth}%` }}>
              <PDFViewerPanel 
                source={selectedSource}
                onClose={() => setSelectedSource(null)}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
