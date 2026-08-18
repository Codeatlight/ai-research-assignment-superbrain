import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Claim & Evidence Research',
  description:
    'Research claim verification / exploration tool: inspect claims and their supporting source evidence.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
