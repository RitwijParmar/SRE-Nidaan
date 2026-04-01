import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SRE निदान | Incident Command Copilot",
  description:
    "SRE निदान helps teams diagnose incidents quickly, review causal evidence, and approve safer interventions in a clear guided workflow.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
