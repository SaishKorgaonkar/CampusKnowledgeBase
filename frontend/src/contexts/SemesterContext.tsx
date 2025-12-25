"use client";

import { createContext, useContext, useState, useEffect } from "react";

type SemesterContextType = {
  course: string;
  semesterNumber: string;
  setCourse: (course: string) => void;
  setSemesterNumber: (semester: string) => void;
  getFullSemester: () => string; // Returns "COMPS-Sem-3" format
};

const SemesterContext = createContext<SemesterContextType | undefined>(undefined);

export const COURSES = [
  { value: "FY", label: "First Year" },
  { value: "COMPS", label: "Computer Engineering" },
  { value: "IT", label: "Information Technology" },
  { value: "AIDS", label: "AI & Data Science" },
  { value: "RAI", label: "Robotics & AI" },
  { value: "EXTC", label: "Electronics & Telecom" },
  { value: "CCE", label: "Computer & Communication" },
  { value: "MECH", label: "Mechanical Engineering" },
  { value: "VLSI", label: "Very Large Scale Industry" },
  { value: "EXCP", label: "Electronics & Computer" },
];

export const SEMESTER_NUMBERS = [
  { value: "1", label: "Semester 1" },
  { value: "2", label: "Semester 2" },
  { value: "3", label: "Semester 3" },
  { value: "4", label: "Semester 4" },
  { value: "5", label: "Semester 5" },
  { value: "6", label: "Semester 6" },
  { value: "7", label: "Semester 7" },
  { value: "8", label: "Semester 8" },
];

export function SemesterProvider({ children }: { children: React.ReactNode }) {
  const [course, setCourseState] = useState<string>("FY");
  const [semesterNumber, setSemesterNumberState] = useState<string>("1");

  useEffect(() => {
    // Load from localStorage on mount
    const savedCourse = localStorage.getItem("selectedCourse");
    const savedSemester = localStorage.getItem("selectedSemesterNumber");
    if (savedCourse) setCourseState(savedCourse);
    if (savedSemester) setSemesterNumberState(savedSemester);
  }, []);

  const setCourse = (newCourse: string) => {
    setCourseState(newCourse);
    localStorage.setItem("selectedCourse", newCourse);
  };

  const setSemesterNumber = (newSemester: string) => {
    setSemesterNumberState(newSemester);
    localStorage.setItem("selectedSemesterNumber", newSemester);
  };

  const getFullSemester = () => {
    return `${course}-Sem-${semesterNumber}`;
  };

  return (
    <SemesterContext.Provider value={{ course, semesterNumber, setCourse, setSemesterNumber, getFullSemester }}>
      {children}
    </SemesterContext.Provider>
  );
}

export function useSemester() {
  const context = useContext(SemesterContext);
  if (context === undefined) {
    throw new Error("useSemester must be used within a SemesterProvider");
  }
  return context;
}
