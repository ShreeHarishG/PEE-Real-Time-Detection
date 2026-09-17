import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EdgeVision | PPE Safety Platform",
  description: "Real-time PPE Compliance and Work-at-Height Safety Monitoring Dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="h-screen overflow-hidden flex flex-col" style={{ background: '#f8f9fc' }}>
        {children}
      </body>
    </html>
  );
}
