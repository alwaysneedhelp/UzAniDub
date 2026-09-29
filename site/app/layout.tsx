import type { Metadata } from "next";
import { headers } from "next/headers";
import "@fontsource-variable/geist";
import SiteHeader from "./components/site-header";
import SiteFooter from "./components/site-footer";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const card = `${protocol}://${host}/og-v2.png`;
  const description = "Uzbek dubbed anime, free to watch.";
  return {
    title: { default: "UzAniDub", template: "%s | UzAniDub" },
    description,
    openGraph: { title: "UzAniDub", description, images: [{ url: card }] },
    twitter: { card: "summary_large_image", title: "UzAniDub", description, images: [card] },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <SiteHeader />
        <main className="site-main">{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
