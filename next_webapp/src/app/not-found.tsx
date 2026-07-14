import Link from "next/link";

export default function NotFound() {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: "sans-serif", backgroundColor: "#fff" }}>
        <div style={{ display: "flex", minHeight: "100vh", flexDirection: "column" }}>
          <header style={{ borderBottom: "1px solid #e5e7eb", padding: "12px 24px" }}>
            <Link href="/en">
              <img src="/img/new-logo-green.png" alt="Melbourne Open Data" style={{ height: "60px" }} />
            </Link>
          </header>
          <main style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "80px 16px", textAlign: "center" }}>
            <div>
              <p style={{ fontSize: "6rem", fontWeight: 800, color: "#16a34a", margin: 0 }}>404</p>
              <h1 style={{ fontSize: "2rem", fontWeight: 700, color: "#111827", marginTop: "16px" }}>Page not found</h1>
              <p style={{ color: "#6b7280", marginTop: "12px" }}>Sorry, we couldn&apos;t find the page you were looking for.</p>
              <div style={{ marginTop: "32px", display: "flex", gap: "12px", justifyContent: "center", flexWrap: "wrap" }}>
                <Link href="/en" style={{ backgroundColor: "#16a34a", color: "#fff", padding: "12px 24px", borderRadius: "12px", textDecoration: "none", fontWeight: 600, fontSize: "14px" }}>Go back home</Link>
                <Link href="/en/contact" style={{ border: "1px solid #d1d5db", color: "#374151", padding: "12px 24px", borderRadius: "12px", textDecoration: "none", fontWeight: 600, fontSize: "14px" }}>Contact support</Link>
              </div>
            </div>
          </main>
          <footer style={{ borderTop: "1px solid #e5e7eb", padding: "24px", textAlign: "center", color: "#9ca3af", fontSize: "14px" }}>
            © {new Date().getFullYear()} Melbourne Open Playground
          </footer>
        </div>
      </body>
    </html>
  );
}