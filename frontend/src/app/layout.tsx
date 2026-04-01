import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SRE-Nidaan | Incident Response Copilot (एसआरई निदान)",
  description:
    "SRE-Nidaan helps teams describe incidents, run grounded causal analysis, approve interventions safely, and capture analyst feedback in one clear workflow.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#f3eee5] text-[#1f2a36] antialiased">{children}</body>
    </html>
  );
}
