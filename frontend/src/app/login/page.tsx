import { Suspense } from "react";
import LoginClient from "./LoginClient";

export const dynamic = "force-dynamic";

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <main className="min-h-screen flex items-center justify-center bg-[#1e1e2e] text-[#cdd6f4]">
          Loading...
        </main>
      }
    >
      <LoginClient />
    </Suspense>
  );
}
