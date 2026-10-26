import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Candidate, JoinClass } from "./Candidate";
import "./index.css";
import { ClassView, Live, Login, Sessions } from "./Proctor";
import { Review } from "./Review";

// Hash routes: #/c/<session>/<token> candidate | #/j/<class>/<token> class link | #/class/<id> class grid | #/login | #/link/<key> one-click login | #/live/<id> | #/review/<id> | anything else: sessions.
// The candidate token stays in the fragment, so it never reaches server logs.
function App() {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const f = () => setHash(location.hash);
    addEventListener("hashchange", f);
    return () => removeEventListener("hashchange", f);
  }, []);
  const [, page, a, b] = hash.split("/");
  if (page === "c" && a && b) return <Candidate key={a} sid={a} token={b} />;
  if (page === "j" && a && b) return <JoinClass cid={a} token={b} />;
  if (page === "class" && a) return <ClassView key={a} cid={a} />;
  if (page === "login") return <Login />;
  if (page === "link" && a) return <Login linkKey={a} />;
  if (page === "live" && a) return <Live key={a} sid={a} />;
  if (page === "review" && a) return <Review key={a} sid={a} />;
  return <Sessions />;
}

createRoot(document.getElementById("root")!).render(<App />);
