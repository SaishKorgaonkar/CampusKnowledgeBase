"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import ChatSidebar from "@/components/chat/ChatSidebar";
import ChatWindow from "@/components/chat/ChatWindow";
import { Session, Message, Source } from "@/types/chat";
import { mockSessions } from "@/lib/mockData";
import { useSemester } from "@/contexts/SemesterContext";

// Load PDFViewerPanel only on client-side to avoid SSR issues with DOMMatrix
const PDFViewerPanel = dynamic(
  () => import("@/components/chat/PDFViewerPanel"),
  { ssr: false }
);

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
  const [isLoading, setIsLoading] = useState(false);

  // Get current session
  const currentSession = sessions.find((s) => s.id === sessionId);

  // Redirect if session doesn't exist
  useEffect(() => {
    if (!currentSession && sessions.length > 0) {
      router.push(`/chat/${sessions[0].id}`);
    }
  }, [currentSession, sessions, router]);

  const handleNewSession = () => {
    const newSession: Session = {
      id: crypto.randomUUID(),
      subject: "New chat",
      messageCount: 0,
      createdAt: new Date(),
    };

    setSessions((prev) => [...prev, newSession]);
    router.push(`/chat/${newSession.id}`);
  };

  const handleSendMessage = async (content: string) => {
    if (!currentSession) return;

    const userMessage: Message = {
      id: `${Date.now()}-user`,
      role: "user",
      content,
      timestamp: new Date(),
    };

    setSessions((prev) =>
      prev.map((s) =>
        s.id === sessionId && s.subject === "New chat"
          ? {
              ...s,
              subject:
                content.length > 32 ? content.slice(0, 32) + "…" : content,
            }
          : s
      )
    );

    // Optimistic UI update (user message immediately)
    const sessionMessages = messages[sessionId] || [];
    setMessages({
      ...messages,
      [sessionId]: [...sessionMessages, userMessage],
    });
    setIsLoading(true);

    try {
      const token = localStorage.getItem("auth_token");
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          question: content,
          course,
          semester: semesterNumber.startsWith("Sem-")
            ? semesterNumber
            : `Sem-${semesterNumber}`,
        }),
      });

      if (!res.ok) {
        throw new Error("Failed to fetch answer");
      }

      const data = await res.json();

      const aiMessage: Message = {
        id: `${Date.now()}-ai`,
        role: "assistant",
        content: data.answer,
        timestamp: new Date(),
        sources: (data.sources || []).map((src: any, index: number) => ({
          id: `${sessionId}-src-${index}`,
          fileName: src.fileName || src.doc_name || "Unknown",
          title: (
            src.title ||
            src.fileName ||
            src.doc_name ||
            "Document"
          ).replace(".pdf", ""),
          pageNumber: src.pageNumber || src.page,
          relevance: src.relevance || 0.85,
          excerpt: src.excerpt || src.text || "",
          filePath: src.filePath || src.source_path || "",
        })),
      };

      setMessages((prev) => ({
        ...prev,
        [sessionId]: [...(prev[sessionId] || []), aiMessage],
      }));

      setSessions((prev) =>
        prev.map((s) =>
          s.id === sessionId ? { ...s, messageCount: s.messageCount + 2 } : s
        )
      );
    } catch (err) {
      console.error(err);

      const errorMessage: Message = {
        id: `${Date.now()}-error`,
        role: "assistant",
        content:
          "Sorry, something went wrong while fetching the answer. Please try again.",
        timestamp: new Date(),
      };

      setMessages((prev) => ({
        ...prev,
        [sessionId]: [...(prev[sessionId] || []), errorMessage],
      }));
    } finally {
      setIsLoading(false);
    }
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
    <div className="flex h-screen w-screen overflow-hidden fixed inset-0">
      <ChatSidebar
        sessions={sessions}
        onNewSession={handleNewSession}
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <div className="flex-1 flex overflow-hidden min-w-0">
        <div
          className="flex flex-col overflow-hidden"
          style={{ width: selectedSource ? `${chatWidth}%` : "100%" }}
        >
          <ChatWindow
            subjectName={currentSession.subject}
            messages={messages[sessionId] || []}
            onSendMessage={handleSendMessage}
            onSourceClick={setSelectedSource}
            isLoading={isLoading}
          />
        </div>
        {selectedSource && (
          <>
            {/* Resizable Divider */}
            <div
              className="w-1 bg-[var(--border-color)] hover:bg-[var(--accent-primary)] cursor-col-resize transition-colors flex-shrink-0"
              onMouseDown={(e) => {
                e.preventDefault();
                const startX = e.clientX;
                const startWidth = chatWidth;

                const handleMouseMove = (e: MouseEvent) => {
                  requestAnimationFrame(() => {
                    const containerWidth =
                      window.innerWidth - (sidebarCollapsed ? 64 : 320);
                    const delta = ((e.clientX - startX) / containerWidth) * 100;
                    const newWidth = Math.max(
                      30,
                      Math.min(70, startWidth + delta)
                    );
                    setChatWidth(newWidth);
                  });
                };

                const handleMouseUp = () => {
                  document.removeEventListener("mousemove", handleMouseMove);
                  document.removeEventListener("mouseup", handleMouseUp);
                  document.body.style.cursor = "";
                  document.body.style.userSelect = "";
                };

                document.body.style.cursor = "col-resize";
                document.body.style.userSelect = "none";
                document.addEventListener("mousemove", handleMouseMove);
                document.addEventListener("mouseup", handleMouseUp);
              }}
            />
            <div
              className="flex flex-col overflow-hidden"
              style={{ width: `${100 - chatWidth}%` }}
            >
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
