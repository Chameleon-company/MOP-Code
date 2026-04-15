import React, { useState, useRef } from 'react';
import Box from '@mui/material/Box'
import { TextField } from '@mui/material'


const MultipleFileUploader = () => {
    const [files, setFiles] = useState<File[]>([]);
    const [loading, setLoading] = useState(false)
    const [uploadSuccess, setUploadSuccess] = useState(false)
    const [returnData, setReturnData] = useState<any[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);


    const clearFiles = () => {
        setFiles([]);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
        setFiles(Array.from(e.target.files));
        setUploadSuccess(false)
        }
    };

    const handleReset = () => {
        setReturnData([]);
        setFiles([]);
        setLoading(false)
        setUploadSuccess(false)
        clearFiles()
    }

    const handleUpload = async () => {
        setReturnData([]);
        if (files) {
            setLoading(true)
            console.log('Uploading file');
            const fileNum = files.length
            for (const file of files) {
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
                    setReturnData(prev => [...prev, data]);
                    console.log(data)
                }
                else {
                    const error = await result.json();
                    console.error('Upload failed:', error);
                    setUploadSuccess(false)
                }

                
                }
                catch (error) {
                    console.error(error)
                }
            }
            setLoading(false)
        }
    };


    return (
        <>
        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3, mt: 6 }}>
            <div className="input-group">
            <input id="file" ref={fileInputRef} type="file" multiple onChange={handleFileChange} disabled={loading} style={{ fontSize: '1.2rem', padding: '12px 20px' }} />
            </div>

            {uploadSuccess && (
                <button 
                    onClick={handleReset}
                    className="submit"
                    disabled={loading}
                    style={{ fontSize: '1.2rem', padding: '12px 20px' }}
                >Upload new files:
                </button>
            )}
            
            

            {files.length > 0 && !uploadSuccess && (
                <>
                <Box sx={{ marginTop: 1, width: '12%', maxWidth: '80%', marginX: 'auto' }}>
                    <TextField
                        fullWidth
                        label="Files to be uploaded"
                        multiline
                        rows={files.length}
                        value={files.map(f => f.name).join('\n')}
                        sx={{
                            '& .MuiInputBase-input': { color: 'white', textAlign: 'center', maxHeight: '350px', overflowY: 'auto !important',},
                            '& .MuiInputLabel-root': { color: 'white' },
                        }}
                    />
                </Box>
                <button 
                    onClick={handleUpload}
                    className="submit"
                    disabled={loading}
                    style={{ fontSize: '1.2rem', padding: '12px 20px' }}
                >Upload files
                </button>
                </>
            )}

                

            {uploadSuccess && returnData && (
            <>
                <div>{returnData.length} reports successfully generated</div>
                <Box sx={{ marginTop: 1, width: '35%', maxWidth: '80%', marginX: 'auto', maxHeight: '900px', overflowY: 'auto !important' }}>
                    {returnData.map((data, index) => (
                    <div key={index}>
                        <h4>Report successfully generated for file {index + 1}:</h4>
                        <table style={{ borderCollapse: 'collapse', width: '100%' }}>
                        <thead>
                            <tr>
                            <th style={{ border: '1px solid #ccc', padding: '8px', textAlign: 'center' }}>Field</th>
                            <th style={{ border: '1px solid #ccc', padding: '8px', textAlign: 'center' }}>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {[
                            ['Image ID', data.image_id],    
                            ['Severity', data.severity],
                            ['Damage Level', data.damage_level],
                            ['Largest Crack Area Ratio', data.largest_crack_area_ratio],
                            ['Largest Crack Est. Length', data.largest_crack_est_length],
                            ['Num Crack Regions', data.num_crack_regions],
                            ['Next Action', data.nextAction],
                            ['Recommended Repair', data.recommendedRepair],
                            ['Risk Management', data.riskManagement],
                            ].map(([label, value]) => (
                            <tr key={label}>
                                <td style={{ border: '1px solid #ccc', padding: '8px', textAlign: 'center' }}>{label}</td>
                                <td style={{ border: '1px solid #ccc', padding: '8px', textAlign: 'center' }}>{String(value)}</td>
                            </tr>
                            ))}
                        </tbody>
                        </table>
                    </div>
                    
                    ))}
                </Box>
            </>
            )}

            
        </Box>
        </>
    );
    };

    export default MultipleFileUploader;