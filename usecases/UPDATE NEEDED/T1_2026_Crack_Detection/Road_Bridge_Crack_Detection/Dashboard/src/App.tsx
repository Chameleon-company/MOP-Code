import "./App.css";
import { Home } from "./Pages/Home";
import { GenerateReports } from "./Pages/generateReports";
import { BatchUpload } from "./Pages/batchUpload";
import { Routes, Route, Navigate } from 'react-router-dom'

function App() {
  return (
    <Routes>
      <Route>
        {/*PUBLIC ROUTE */}
        <Route path="/" element={<Navigate to="/home" replace />} />
        <Route path="/home" element={<Home />} />
        <Route path="/Uploadmultiplefiles" element={<BatchUpload />} />
        <Route path="/generateReports" element={<GenerateReports />} />

      </Route>
    </Routes>
  );
}

export default App;
