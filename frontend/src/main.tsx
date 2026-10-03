import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Candidate } from "./Candidate";
import "./index.css";
import { Live, Login, Sessions } from "./Proctor";
import { Review } from "./Review";

// Hash routes: #/c/<session>/<token> candidate | #/login | #/live/<id> | #/review/<id> | anything else: sessions.
// The candidate token stays in the fragment, so it never reaches server logs.
function App() {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const f = () => setHash(location.hash);
    addEventListener("hashchange", f);
    return () => removeEventListener("hashchange", f);
  }, []);
  const [, page, a, b] = hash.split("/");
  if (page === "c" && a && b) return <Candidate sid={a} token={b} />;
  if (page === "login") return <Login />;
  if (page === "live" && a) return <Live key={a} sid={a} />;
  if (page === "review" && a) return <Review key={a} sid={a} />;
  return <Sessions />;
}

createRoot(document.getElementById("root")!).render(<App />);
