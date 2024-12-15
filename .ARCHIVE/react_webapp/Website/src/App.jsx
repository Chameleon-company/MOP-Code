import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Home from "./pages/Home";
// import Contact from "./pages/Contact";
import About from "./pages/About";
import "./App.css";
import Casestudies from "./pages/Casestudies";
import Datasets from "./pages/Datasets";
import Privacypolicy from "./pages/PrivacyPolicy";
import Licensing from "./pages/Licensing/Licensing";
import SearchResults from "./pages/SearchResult";
import Contact from "./pages/Contact";


function App() {
  return (
    <main>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          {/* <Route path="/contact" element={<Contact />} /> */}
          <Route path="/casestudies" element={<Casestudies />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/privacypolicy" element={<Privacypolicy />} />
          <Route path="/licensing" element={<Licensing />} />
          <Route path="/searchresults" element={<SearchResults />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </BrowserRouter>
    </main>
  );
}

export default App;
