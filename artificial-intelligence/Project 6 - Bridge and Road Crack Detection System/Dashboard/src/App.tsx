import "./App.css";
import { Home } from "./Pages/Home";
import { Uploader } from "./Pages/upload";
import { BatchUpload } from "./Pages/batchUpload";
import { Routes, Route, Navigate } from 'react-router-dom'

function App() {
  return (
    <Routes>
      <Route>
        {/*PUBLIC ROUTE */}
        <Route path="/" element={<Navigate to="/home" replace />} />
        <Route path="/home" element={<Home />} />
        <Route path="/uploader" element={<Uploader />} />
        <Route path="/Uploadmultiplefiles" element={<BatchUpload />} />

      </Route>
    </Routes>
  );
}

export default App;
