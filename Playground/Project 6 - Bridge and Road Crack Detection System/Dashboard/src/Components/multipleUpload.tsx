import React, { useState, useRef } from 'react';
import { Box, Typography } from '@mui/material';
import { TextField } from '@mui/material'
import {SeverityBadge} from './severityBadge'

const MultipleFileUploader = () => {
    const [files, setFiles] = useState<File[]>([]);
    const [loading, setLoading] = useState(false)
    const [uploadSuccess, setUploadSuccess] = useState(false)
    const [returnData, setReturnData] = useState<any[]>([])
    const fileInputRef = useRef<HTMLInputElement>(null)
    const [checked, setChecked] = useState(false)


    const clearFiles = () => {
        setFiles([]);
        if (fileInputRef.current) fileInputRef.current.value = ''
    }

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
        setFiles(Array.from(e.target.files));
        setUploadSuccess(false)
        }
    }

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
            for (const file of files) {
                const formData = new FormData()
                formData.append('file', file)
                
                let url = ''
                if (checked){
                    url = `${(import.meta as any).env.VITE_API_URL}/api/uploadImage?flag=true`
                }
                else{
                    url = `${(import.meta as any).env.VITE_API_URL}/api/uploadImage?flag=false`
                }
                console.log(url)

                try {
                    const _url = url
                    const result = await fetch(_url, {
                        method: 'POST',
                        body: formData
                    })
                
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
    }


    return (
        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3, mt: 6 }}>
            
            {/* ── Upload Files button ────────────── */}
            {!uploadSuccess && (
                <Box sx={{ display: "flex", flexDirection: "row", alignItems: "center", gap: 3, mt: 6 }}>
                    <Box
                        onClick={() => fileInputRef.current?.click()}
                        sx={{
                            border: '1.5px dashed',
                            borderColor: files.length > 0 ? 'rgba(99,153,34,0.55)' : 'rgba(255,255,255,0.12)',
                            borderRadius: '12px',
                            bgcolor: files.length > 0 ? 'rgba(99,153,34,0.04)' : 'rgba(255,255,255,0.02)',
                            py: 6, px: 4,
                            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1.5,
                            cursor: loading ? 'not-allowed' : 'pointer',
                            transition: 'all 0.2s ease',
                            '&:hover': !loading ? {
                                borderColor: 'rgba(99,153,34,0.6)',
                                bgcolor: 'rgba(99,153,34,0.06)',
                            } : {},
                        }}
                    >
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
                            stroke={files.length > 0 ? '#7ec83a' : 'rgba(255,255,255,0.25)'} strokeWidth="1.2">
                            <polyline points="16 16 12 12 8 16"/>
                            <line x1="12" y1="12" x2="12" y2="21"/>
                            <path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3"/>
                        </svg>
                        <Typography sx={{ fontSize: '0.9rem', color: files.length > 0 ? '#a8d870' : 'rgba(255,255,255,0.35)' }}>

                        {files.length > 0
                            ? `${files.length} file${files.length > 1 ? 's' : ''} selected`
                            : 'Click to select files'}

                        </Typography>
                        <Typography sx={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace' }}>
                            .png · .jpg · .jpeg
                        </Typography>


                        <input
                            id="file"
                            ref={fileInputRef}
                            type="file"
                            multiple
                            onChange={handleFileChange}
                            disabled={loading}
                            style={{ display: 'none' }}
                        />
                    </Box>
                    {/* ── Files to be uploaded ────────────── */}
                    <Box sx={{ width: 700, height: 200 }}>
                        <Box sx={{ marginTop: 1, width: '100%', maxWidth: '100%', height: '250px !important', maxHeight: '100%', marginX: 'auto'}}>
                            <TextField
                                fullWidth
                                label="Files to be uploaded"
                                multiline
                                rows={6}
                                value={files.map(f => f.name).join('\n')}
                                InputLabelProps={{ shrink: true }}
                                InputProps={{ readOnly: true }}
                                sx={{
                                    '& .MuiInputBase-input': { color: 'white', textAlign: 'center', maxHeight: '350px', overflowY: 'auto !important',},
                                    '& .MuiInputLabel-root': { color: 'white' }
                                }}
                            />
                        </Box>
                    </Box>
                </Box>
                
            )}  

            {/* ── Button display after upload ────────────── */}
            {uploadSuccess && (
                <button 
                    onClick={handleReset}
                    className="submit"
                    disabled={loading}
                    style={{ fontSize: '1.2rem', padding: '12px 20px', borderRadius: '12px' }}
                >Upload new files:
                </button>
            )}
            
            
            
            {files.length > 0 && !uploadSuccess && (
                <>
                {/* ── Upload Button ────────────── */}
                <button 
                    onClick={handleUpload}
                    className="submit"
                    disabled={loading}
                    style={{ fontSize: '1.2rem', padding: '12px 20px', borderRadius: '12px' }}
                >Upload files
                </button>


                {/* ── Generate AI report button ────────────── */}
                <label style={{ fontSize: '0.82rem', fontFamily: 'monospace', color: 'rgba(255,255,255,0.92)' }}>
                <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => setChecked(e.target.checked)}
                    disabled={loading}
                />
                Generate AI Report?
                </label>
                <p style={{ fontSize: "0.65rem", fontFamily: 'monospace', color: 'rgba(255, 255, 255, 0.92)'  }}>Each report takes around 20 seconds to generate. This can also be done later in the Generate Reports Tab</p>
                </>
            )}

                

            {uploadSuccess && returnData && (
            <>
                <Typography sx={{ fontSize: '0.82rem', fontFamily: 'monospace', color: 'rgba(255,255,255,0.5)' }}>
                    {returnData.length} reports successfully generated
                </Typography>

                <Box sx={{ marginTop: 1, width: '55%', maxWidth: '80%', marginX: 'auto', maxHeight: '900px', overflowY: 'auto !important' }}>
                    {returnData.map((data, index) => (
                    <div key={index}>
                        <Box sx={{ minHeight: '20vh', bgcolor: '#141720', color: '#e8eaf0', borderRadius: '12px', marginTop: 2 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                            <Box sx={{
                                width: 40, height: 40, borderRadius: '50%',
                                bgcolor: 'rgba(99,153,34,0.2)',
                                border: '1px solid rgba(99,153,34,0.4)',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: '0.65rem', fontWeight: 700, color: '#7ec83a',
                            }}>
                                {index + 1}
                            </Box>
                            
                            <Typography sx={{ fontSize: '0.82rem', fontFamily: 'monospace', color: 'rgba(255,255,255,0.5)' }}>
                                Report for: {data.image_id}
                            </Typography>

                            <SeverityBadge severity={data.severity} />
                        </Box>
                        
                        <Box sx={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0,}}>
                            {[
                            ['Crack Severity', data.severity],
                            ['Damage level', data.damage_level],
                            ['Crack area ratio', data.largest_crack_area_ratio],
                            ['Est. crack length', data.largest_crack_length],
                            ['Crack regions', data.num_crack_regions],
                            ['Report status', data.report_status],
                            ['Inspection schedule', data.inspection_schedule],
                            ['Risk assessment', data.risk_assessment],
                            ['Repair actions', data.repair_actions],
                            ].map(([label, value], i) => (
                            <Box key={label as string} sx={{
                                px: 2.5, py: 1.5,
                                borderBottom: i < 6 ? '1px solid rgba(255,255,255,0.05)' : 'none',
                                borderRight: i % 2 === 0 ? '1px solid rgba(255,255,255,0.05)' : 'none',
                            }}>
                                <Typography sx={{ fontSize: '0.68rem', color: 'rgba(255,255,255,0.28)', mb: 0.4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                {label}
                                </Typography>
                                <Typography sx={{ fontSize: '0.85rem', color: 'rgba(255,255,255,0.8)', fontFamily: 'monospace' }}>
                                {String(value)}
                                </Typography>
                            </Box>
                            ))}
                        </Box>
                                  
                    </Box>
                    </div>
                    ))}
                </Box>
            </>
            )}   
        </Box>
    );
    };

    export default MultipleFileUploader;