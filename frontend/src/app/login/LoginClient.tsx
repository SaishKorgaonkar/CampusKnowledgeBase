"use client";

import Image from "next/image";
import { useSearchParams } from "next/navigation";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function LoginClient() {
  const searchParams = useSearchParams();
  const error = searchParams.get("error");

  const handleGoogleLogin = () => {
    window.location.href = `${BACKEND_URL}/login/google`;
  };

  return (
    <main className="min-h-screen flex items-center justify-center bg-[#1e1e2e] text-[#cdd6f4]">
      <div className="w-full max-w-xl mx-4 rounded-3xl border border-[#313244] bg-[#181825] p-10 flex flex-col items-center gap-6">
        <h1 className="text-2xl font-semibold">
          Sign in to Campus Knowledge Base
        </h1>

        <p className="text-sm text-[#a6adc8] text-center">
          Login using your college Google account to access verified campus
          information.
        </p>

        <div className="flex flex-col gap-4 mt-6 w-full max-w-xs">
          {error === "unauthorized_email" && (
            <div className="w-full max-w-xs rounded-xl border border-red-400/40 bg-red-400/10 px-4 py-2 text-sm text-red-300 text-center">
              ❌ Only college email IDs are allowed
            </div>
          )}
          <button
            onClick={handleGoogleLogin}
            className="inline-flex items-center justify-center gap-3 rounded-full border border-[#45475a] bg-[#11111b] px-5 py-2.5 text-sm font-medium text-[#cdd6f4] hover:bg-[#181825] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#89b4fa] transition-colors"
          >
            <span className="inline-flex h-6 w-6 items-center justify-center bg-white rounded-full">
              <Image
                src="/icon-google.svg"
                alt="Google"
                width={16}
                height={16}
              />
            </span>
            <span>Sign in with Somaiya Email</span>
          </button>

          <p className="text-xs text-[#6c7086] text-center">
            Only <span className="font-medium">@somaiya.edu</span> accounts are
            allowed
          </p>
        </div>
      </div>
    </main>
  );
}
