import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SRE-Nidaan | NEXUS-CAUSAL v3.1 — Causal AI Incident Response",
  description:
    "Enterprise AI SRE dashboard using Pearl's causal hierarchy to prevent panic scaling in cloud-native environments. Powered by NEXUS-CAUSAL v3.1.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-nidaan-bg antialiased">{children}</body>
    </html>
  );
}
