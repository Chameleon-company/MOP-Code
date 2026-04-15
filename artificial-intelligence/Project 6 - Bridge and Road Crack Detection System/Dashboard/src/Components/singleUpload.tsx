import React, { useState } from 'react';
import Box from '@mui/material/Box'

const FileUploader = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [returnData, setReturnData] = useState<any>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
      setUploadSuccess(false)
    }
  };

  const handleUpload = async () => {
    if (file) {
      setLoading(true)
      console.log('Uploading file');

      const formData = new FormData();
      formData.append('file', file);

      try {
        const url = `${(import.meta as any).env.VITE_API_URL}/api/uploadImage`
        const result = await fetch(url, {
          method: 'POST',
          body: formData
        });

        
        if (result.ok) {
          setUploadSuccess(true)
          const data = await result.json();
          setReturnData(data)
          console.log(data)
          setLoading(false)
        }
        else {
          const error = await result.json();
          console.error('Upload failed:', error);
          setUploadSuccess(false)
          setLoading(false)
          return;
        }

        
      }
      catch (error) {
        console.error(error)
      }
    }
  };

  return (
    <>
      <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3, mt: 6 }}>
        <div className="input-group">
          <input id="file" type="file" onChange={handleFileChange} disabled={loading} style={{ fontSize: '1.2rem', padding: '12px 20px' }} />
        </div>
        {file && (
          <section>
            File details:
            <ul>
              <li>Name: {file.name}</li>
              <li>Type: {file.type}</li>
              <li>Size: {file.size} bytes</li>
            </ul>
          </section>
        )}

        {file && !uploadSuccess && (
          <button 
            onClick={handleUpload}
            className="submit"
            disabled={loading}
            style={{ fontSize: '1.2rem', padding: '12px 20px' }}
          >Upload file</button>
        )}

        {uploadSuccess && returnData &&(
          <Box>
            <h4>Report successfully generated:</h4>            
            <table style={{ borderCollapse: 'collapse', width: '100%' }}>
            <thead>
              <tr>
                <th style={{ border: '1px solid #ccc', padding: '8px', textAlign: 'center' }}>Field</th>
                <th style={{ border: '1px solid #ccc', padding: '8px', textAlign: 'center' }}>Value</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Severity', returnData.severity],
                ['Damage Level', returnData.damage_level],
                ['Largest Crack Area Ratio', returnData.largest_crack_area_ratio],
                ['Largest Crack Est. Length', returnData.largest_crack_est_length],
                ['Num Crack Regions', returnData.num_crack_regions],
                ['Next Action', returnData.nextAction],
                ['Recommended Repair', returnData.recommendedRepair],
                ['Risk Management', returnData.riskManagement],
              ].map(([label, value]) => (
                <tr key={label}>
                  <td style={{ border: '1px solid #ccc', padding: '8px' }}>{label}</td>
                  <td style={{ border: '1px solid #ccc', padding: '8px' }}>{String(value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </Box>
        )}

        
      </Box>
    </>
  );
};

export default FileUploader;