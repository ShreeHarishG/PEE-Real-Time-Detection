'use client';
import { useEffect, useState, useRef } from 'react';
import { Activity, AlertTriangle, ShieldCheck, Users, Clock, Zap, Upload, Play, Square, Map, Save, X, ChevronRight, Shield, Eye, Check, ArrowRight, Camera, FileText, Video, Trash2 } from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer, Tooltip, XAxis, YAxis, PieChart, Pie, Cell } from 'recharts';

type ModalType = 'fps' | 'violations' | 'resolution' | 'workers' | 'training' | 'cameras' | 'reports' | null;

export default function LiveMonitoring() {
  // States
  const [stats, setStats] = useState<any>(null);
  const [violations, setViolations] = useState<any[]>([]);
  const [allViolations, setAllViolations] = useState<any[]>([]);
  const [fpsHistory, setFpsHistory] = useState<{t: string, fps: number}[]>([]);
  
  // Upload & Jobs
  const [uploadStatus, setUploadStatus] = useState<string>('IDLE');
  const [uploadedVideo, setUploadedVideo] = useState<any>(null);
  const [activeJob, setActiveJob] = useState<any>(null);
  const [jobProgress, setJobProgress] = useState<{progress: number, total: number, fps: number, workers_detected: number, violations_detected: number} | null>(null);
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [jobViolations, setJobViolations] = useState<any[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Zones & Cameras
  const [isEditingZone, setIsEditingZone] = useState(true);
  const [polygonPoints, setPolygonPoints] = useState<number[][]>([]);
  const [zoneConfig, setZoneConfig] = useState<{required: string[]}>({ required: ['helmet', 'vest', 'boots', 'harness'] });
  const [cameras, setCameras] = useState<any[]>([]);
  const [newCamName, setNewCamName] = useState('');
  const [newCamUrl, setNewCamUrl] = useState('');

  // Modal
  const [activeModal, setActiveModal] = useState<ModalType>(null);
  
  // Training Feedback State
  const [trainingQueue, setTrainingQueue] = useState<any[]>([]);
  const [currentTrainingIndex, setCurrentTrainingIndex] = useState(0);
  const [feedback, setFeedback] = useState({ correct: true, helmet: false, vest: false, boots: false, harness: false });

  const fileInputRef = useRef<HTMLInputElement>(null);

  // No longer loading state from localStorage on mount so that the app always starts fresh

  // No longer saving state to localStorage when it changes

  // Data Fetching
  useEffect(() => {
    const fetchStatsAndEvents = async () => {
      try {
        const statsRes = await fetch('/api/v1/stats');
        if (statsRes.ok) setStats(await statsRes.json());

        const eventsRes = await fetch('/api/v1/violations?limit=5');
        if (eventsRes.ok) setViolations(await eventsRes.json());
        
        const camRes = await fetch('/api/v1/cameras');
        if (camRes.ok) setCameras(await camRes.json());
      } catch (e) {
        // Silently handle fetch failures during backend reloads/restarts
        console.debug("Transient fetch error during polling (backend may be reloading):", e);
      }
    };

    fetchStatsAndEvents();
    const interval = setInterval(fetchStatsAndEvents, 2000);
    return () => clearInterval(interval);
  }, []);

  // Fetch all violations when modal opens
  useEffect(() => {
    if (activeModal === 'violations' || activeModal === 'workers' || activeModal === 'resolution' || activeModal === 'training') {
      const fetchAll = async () => {
        try {
          const res = await fetch('/api/v1/violations?limit=500');
          if (res.ok) {
            const data = await res.json();
            setAllViolations(data);
            if (activeModal === 'training') {
              // Only queue items that haven't received feedback
              const queue = data.filter((v: any) => v.feedback_correct === null);
              setTrainingQueue(queue);
              setCurrentTrainingIndex(0);
              setFeedback({ correct: true, helmet: false, vest: false, boots: false, harness: false });
            }
          }
        } catch (e) {}
      };
      fetchAll();
    }
  }, [activeModal]);

  // Fetch initial zone data
  useEffect(() => {
    const fetchZone = async () => {
      try {
        const res = await fetch('/api/v1/zones');
        if (res.ok) {
          const zones = await res.json();
          const activeZone = zones.find((z: any) => z.id === 1) || zones[0];
          if (activeZone) {
            setZoneConfig({ required: activeZone.required_ppe || ['helmet', 'vest', 'boots', 'harness'] });
            if (activeZone.polygon) setPolygonPoints(activeZone.polygon);
          }
        }
      } catch(e) {}
    };
    fetchZone();
  }, []);

  // Job Polling
  useEffect(() => {
    if (!activeJob || activeJob.status === 'stopped' || activeJob.status === 'completed') return;
    const pollJob = async () => {
      try {
        const res = await fetch(`/api/v1/jobs/${activeJob.id}`);
        if (res.ok) {
          const data = await res.json();
          setJobProgress({ 
            progress: data.progress, 
            total: data.total_frames, 
            fps: data.fps,
            workers_detected: data.workers_detected || 0,
            violations_detected: data.violations_detected || 0
          });
          if (data.status !== activeJob.status) {
            setActiveJob(data);
            if (data.status === 'processing') {
              setStreamUrl(`/api/v1/stream?jobId=${data.id}`);
            }
          }
          if (data.fps > 0) {
            setFpsHistory(prev => [...prev, { t: new Date().toLocaleTimeString(), fps: data.fps }].slice(-30));
          }
        }
      } catch(e) {}
    };
    const interval = setInterval(pollJob, 1000);
    return () => clearInterval(interval);
  }, [activeJob]);

  // Fetch job violations when completed
  useEffect(() => {
    if (activeJob && activeJob.status === 'completed') {
      const fetchJobViolations = async () => {
        try {
          const res = await fetch(`/api/v1/jobs/${activeJob.id}/violations`);
          if (res.ok) setJobViolations(await res.json());
        } catch(e) {}
      };
      fetchJobViolations();
    }
  }, [activeJob]);

  // Actions
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setUploadStatus('UPLOADING');
      
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      try {
        const res = await fetch('/api/v1/videos/upload', {
          method: 'POST',
          body: formData
        });
        if (res.ok) {
          const data = await res.json();
          setUploadedVideo(data);
          setActiveJob(null);
          setUploadStatus('SUCCESS');
        } else {
          setUploadStatus('FAILED');
        }
      } catch (err) {
        setUploadStatus('FAILED');
      }
    }
  };

  const loadDemoVideo = (filename: string) => {
    setUploadStatus('DEMO_SELECTED');
    setUploadedVideo({ id: `demo_${filename.replace('.mp4', '')}`, filepath: `../docs/${filename}` });
    setActiveJob(null);
  };

  const startProcessing = async () => {
    if (!uploadedVideo) return;
    try {
      // Always save the zone config (including PPE requirements) before processing.
      // Polygon points will just be empty if not drawn, which is safely handled.
      await saveZone();
      
      const endpoint = uploadedVideo.isCamera 
        ? `/api/v1/cameras/${uploadedVideo.id}/process`
        : `/api/v1/jobs/${uploadedVideo.id}/process`;
        
      const res = await fetch(endpoint, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setActiveJob(data);
        setJobProgress({ progress: 0, total: 0, fps: 0, workers_detected: 0, violations_detected: 0 });
        setJobViolations([]);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const stopProcessing = async () => {
    if (!activeJob) return;
    try {
      await fetch(`/api/v1/jobs/${activeJob.id}/stop`, { method: 'POST' });
      setActiveJob({ ...activeJob, status: 'stopped' });
    } catch(e) {}
  };

  const saveZone = async () => {
    try {
      await fetch('/api/v1/zones/1', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ polygon: polygonPoints, required_ppe: zoneConfig.required })
      });
      // Keeping isEditingZone true so they can keep seeing/editing it before they hit start.
    } catch(e) {}
  };

  const handleVideoClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isEditingZone) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setPolygonPoints([...polygonPoints, [x, y]]);
  };

  const clearPolygon = () => {
    setPolygonPoints([]);
  };

  const togglePPE = (item: string) => {
    setZoneConfig(prev => {
      const req = prev.required.includes(item)
        ? prev.required.filter(i => i !== item)
        : [...prev.required, item];
      return { required: req };
    });
  };

  const acknowledgeViolation = async (id: number) => {
    try {
      await fetch(`/api/v1/violations/${id}/acknowledge`, { method: 'PATCH' });
      setViolations(prev => prev.map(v => v.id === id ? { ...v, acknowledged: true } : v));
      setAllViolations(prev => prev.map(v => v.id === id ? { ...v, acknowledged: true } : v));
    } catch (e) {}
  };

  const submitTrainingFeedback = async (id: number) => {
    try {
      await fetch(`/api/v1/violations/${id}/feedback`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          feedback_correct: feedback.correct,
          feedback_helmet: feedback.helmet,
          feedback_vest: feedback.vest,
          feedback_boots: feedback.boots,
          feedback_harness: feedback.harness
        })
      });
      
      // Update local state to reflect it's acknowledged/feedback provided
      setAllViolations(prev => prev.map(v => v.id === id ? { ...v, acknowledged: true, feedback_correct: feedback.correct } : v));
      setViolations(prev => prev.map(v => v.id === id ? { ...v, acknowledged: true, feedback_correct: feedback.correct } : v));
      
      // Move to next in queue
      if (currentTrainingIndex < trainingQueue.length - 1) {
        setCurrentTrainingIndex(prev => prev + 1);
        setFeedback({ correct: true, helmet: false, vest: false, boots: false, harness: false });
      } else {
        // Queue finished
        setTrainingQueue([]);
      }
    } catch (e) {}
  };

  const seekToViolation = (timestampSec: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timestampSec;
      videoRef.current.play();
    }
  };

  // Helpers
  const displayFps = jobProgress?.fps ? jobProgress.fps.toFixed(1) : (stats?.latest_fps ? Number(stats.latest_fps).toFixed(1) : '0.0');
  const activeCount = stats?.active_violations || 0;
  const resolutionRate = stats?.resolution_rate || 0;

  const jobState = activeJob ? activeJob.status : (uploadedVideo ? 'configuration' : 'empty');

  const getEvidenceUrl = (path: string) => {
    if (!path) return null;
    const parts = path.split(/[/\\]evidence[/\\]/);
    if (parts.length > 1) {
      return `/evidence/${parts[1].replace(/\\/g, '/')}`;
    }
    const filename = path.split(/[/\\]/).pop();
    return `/evidence/${filename}`;
  };

  // Workers grouped data
  const workerData = allViolations.reduce((acc: any, v: any) => {
    const wid = v.worker_tracking_id;
    if (!acc[wid]) acc[wid] = { id: wid, violations: [], count: 0 };
    acc[wid].violations.push(v);
    acc[wid].count += 1;
    return acc;
  }, {});

  // Donut data
  const donutData = [
    { name: 'Resolved', value: stats?.acknowledged || 0, color: '#10b981' },
    { name: 'Active', value: stats?.active_violations || 0, color: '#ef4444' },
  ];

  // ======================== RENDER ========================

  return (
    <div className="flex flex-col h-screen bg-slate-50 text-slate-800">
      {/* Top Navbar */}
      <div className="top-navbar shrink-0 bg-white border-b border-slate-200">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #4f46e5, #7c3aed)' }}>
          <Video className="h-4 w-4 text-white" />
        </div>
        <span className="text-base font-bold text-slate-800 tracking-tight">EdgeVision</span>
        <span className="text-[10px] font-semibold text-indigo-500/80 uppercase tracking-widest">Video Analysis</span>
        
        <div className="flex-1" />
        
        <a 
          href="/live"
          className="mr-3 px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all bg-slate-100 hover:bg-slate-200 text-slate-700"
        >
          <Camera className="h-3 w-3" /> Live Dashboard
        </a>

        <button 
          onClick={() => setActiveModal('reports')}
          className="mr-3 px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all bg-slate-100 hover:bg-slate-200 text-slate-700"
        >
          <FileText className="h-3 w-3" /> Reports
        </button>
        
        <button 
          onClick={() => setActiveModal('training')}
          className="mr-4 px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all"
          style={{ background: '#3b82f6', color: '#fff', border: '1px solid #2563eb' }}
        >
          <Zap className="h-3 w-3" /> Model Training
        </button>
        
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg" style={{ background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.15)' }}>
          <div className="glow-dot glow-dot-green" />
          <span className="text-xs font-medium text-emerald-600">System Online</span>
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <div className="py-6">
          <div className="mx-auto max-w-[1400px] px-6 md:px-8">
            
            <div className="mb-6 flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-slate-800 tracking-tight">Video Analysis Workspace</h1>
                <p className="text-sm text-slate-500 mt-1">Upload video files for retroactive PPE compliance scanning.</p>
              </div>
              
              <div className="flex gap-3">
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  accept=".mp4,.avi,.mov,.mkv"
                  onChange={handleUpload}
                />
                {uploadStatus !== 'IDLE' && (
                  <span className="text-xs font-medium px-2 py-1 rounded bg-indigo-50 text-indigo-600 border border-indigo-100 mt-2">
                    {uploadStatus}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-3">
                {uploadedVideo && (!activeJob || activeJob.status === 'stopped' || activeJob.status === 'completed') && (
                  <button className="btn-primary text-sm flex items-center gap-2" onClick={startProcessing}>
                    <Play className="h-4 w-4" /> Start Processing
                  </button>
                )}
                {activeJob && activeJob.status === 'processing' && (
                  <button className="btn-danger text-sm flex items-center gap-2" onClick={stopProcessing}>
                    <Square className="h-4 w-4" /> Stop
                  </button>
                )}
                {activeJob && (
                  <div className={`badge ${activeJob.status === 'processing' ? 'badge-success' : 'badge-warning'}`}>
                    <div className={`glow-dot ${activeJob.status === 'processing' ? 'glow-dot-green' : 'glow-dot-amber'}`} />
                    Job: {activeJob.status.toUpperCase()}
                    {activeJob.status === 'processing' && jobProgress && jobProgress.total > 0 && (
                      <span className="ml-1">({Math.round((jobProgress.progress / jobProgress.total) * 100)}%)</span>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Stats Row — Clickable Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <div className="stat-card stat-card-blue animate-fade-in-delay-1" onClick={() => setActiveModal('fps')}>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">Video FPS</p>
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(79, 70, 229, 0.08)' }}>
                    <Zap className="h-4 w-4" style={{ color: '#4f46e5' }} />
                  </div>
                </div>
                <p className="text-3xl font-bold text-slate-800">{displayFps}</p>
                <div className="view-details"><span>View Details</span><ChevronRight className="h-3 w-3" /></div>
              </div>

              <div className="stat-card stat-card-red animate-fade-in-delay-2" onClick={() => setActiveModal('violations')}>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Active Violations</p>
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(239, 68, 68, 0.1)' }}>
                    <AlertTriangle className="h-4 w-4 text-red-500" />
                  </div>
                </div>
                <p className="text-3xl font-bold text-red-500">{activeCount}</p>
                <div className="view-details"><span>View All</span><ChevronRight className="h-3 w-3" /></div>
              </div>

              <div className="stat-card stat-card-green animate-fade-in-delay-3" onClick={() => setActiveModal('resolution')}>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Resolution Rate</p>
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(16, 185, 129, 0.1)' }}>
                    <ShieldCheck className="h-4 w-4 text-emerald-500" />
                  </div>
                </div>
                <p className="text-3xl font-bold text-emerald-500">{resolutionRate}%</p>
                <div className="view-details"><span>View Breakdown</span><ChevronRight className="h-3 w-3" /></div>
              </div>

              <div className="stat-card stat-card-purple animate-fade-in-delay-4" onClick={() => setActiveModal('workers')}>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Workers Tracked</p>
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
                    <Users className="h-4 w-4 text-purple-500" />
                  </div>
                </div>
                <p className="text-3xl font-bold text-slate-800">{stats?.unique_workers || 0}</p>
                <div className="view-details"><span>View Workers</span><ChevronRight className="h-3 w-3" /></div>
              </div>
            </div>

            {/* Main Content Grid */}
            {jobState === 'empty' && (
              <div className="glass-card p-16 text-center flex flex-col items-center justify-center animate-fade-in min-h-[500px] border-dashed border-2 border-gray-200 bg-white">
                <div className="w-24 h-24 rounded-full flex items-center justify-center mb-6" style={{ background: 'rgba(79, 70, 229, 0.06)' }}>
                  <Upload className="h-10 w-10" style={{ color: '#4f46e5' }} />
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-3">Upload Video File</h2>
                <p className="text-gray-500 max-w-md mb-8 text-base">Select a pre-recorded CCTV video file to analyze for PPE compliance, or use a demo recording.</p>
                <div className="flex flex-col sm:flex-row gap-4 w-full max-w-md justify-center">
                  <button className="btn-primary flex-1 py-3 text-base flex justify-center items-center" onClick={() => fileInputRef.current?.click()}>
                    <Upload className="h-5 w-5 mr-2" /> Upload .MP4
                  </button>
                  <select 
                    className="btn-secondary flex-1 py-3 text-base text-center cursor-pointer border-slate-300 hover:border-indigo-300 appearance-none bg-white"
                    onChange={(e) => loadDemoVideo(e.target.value)}
                    defaultValue=""
                  >
                    <option value="" disabled>Or Select Demo...</option>
                    <option value="test.mp4">test.mp4</option>
                    <option value="test1.mp4">test1.mp4</option>
                    <option value="test2.mp4">test2.mp4</option>
                    <option value="test3.mp4">test3.mp4</option>
                    <option value="vidssave.mp4">Viral Video</option>
                  </select>
                </div>
              </div>
            )}


            {jobState === 'configuration' && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in">
                <div className="lg:col-span-2 glass-card p-4 flex flex-col min-h-[400px] bg-slate-50 border-slate-200">
                  <div className="flex justify-between items-center mb-4">
                    <div>
                      <h3 className="text-lg font-bold text-slate-700 flex items-center gap-2">
                        <Map className="h-5 w-5 text-blue-500" /> Draw Safety Zone
                      </h3>
                      <p className="text-xs text-slate-500 mt-1">Click points on the image below to draw a custom shape (polygon). The AI will ONLY track workers inside this zone!</p>
                    </div>
                    {polygonPoints.length > 0 && (
                      <button onClick={clearPolygon} className="text-xs font-semibold px-3 py-1.5 rounded border border-slate-200 hover:bg-slate-100 text-slate-600 transition-colors">
                        Clear Zone
                      </button>
                    )}
                  </div>
                  
                  <div className="w-full bg-black rounded-lg overflow-hidden relative cursor-crosshair border border-slate-200" style={{ aspectRatio: '16/9' }}>
                    <img 
                      src={uploadedVideo?.isCamera 
                        ? `/api/v1/cameras/${uploadedVideo?.id}/snapshot`
                        : `/api/v1/videos/${uploadedVideo?.id}/snapshot`}
                      alt="First Frame"
                      className="absolute inset-0 w-full h-full object-contain select-none pointer-events-none"
                    />
                    <svg
                      className="absolute inset-0 w-full h-full z-10"
                      onClick={handleVideoClick}
                    >
                      {polygonPoints.length > 0 && (
                        <polygon
                          points={polygonPoints.map(p => `${p[0] * 100}% ${p[1] * 100}%`).join(', ')}
                          fill="rgba(34, 197, 94, 0.2)"
                          stroke="#22c55e"
                          strokeWidth="2"
                          strokeDasharray="4 2"
                        />
                      )}
                      {polygonPoints.map((p, i) => (
                        <circle
                          key={i}
                          cx={`${p[0] * 100}%`}
                          cy={`${p[1] * 100}%`}
                          r="4"
                          fill="#ffffff"
                          stroke="#22c55e"
                          strokeWidth="2"
                        />
                      ))}
                    </svg>
                  </div>
                  
                  <div className="mt-4 flex justify-end">
                    <button className="btn-primary flex items-center gap-2 px-6 py-2" onClick={startProcessing}>
                      <Play className="h-4 w-4" /> Save & Start Processing
                    </button>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="glass-card p-5 border border-indigo-100 shadow-sm shadow-indigo-100/50">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4">Model Configuration</h3>
                    <div className="p-3 rounded-lg flex items-center justify-between mb-4" style={{ background: 'rgba(79, 70, 229, 0.04)', border: '1px solid rgba(79, 70, 229, 0.1)' }}>
                      <span className="text-sm font-medium text-gray-700">Pipeline</span>
                      <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ background: 'rgba(79, 70, 229, 0.1)', color: '#4338ca' }}>V6-HITL OFFLINE</span>
                    </div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 mt-5">PPE Checklist</h3>
                    <div className="space-y-3">
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input type="checkbox" checked={zoneConfig.required.includes('helmet')} onChange={() => togglePPE('helmet')} className="rounded border-slate-300 accent-indigo-600" />
                        <span className="text-sm text-slate-700">Helmet Required</span>
                      </label>
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input type="checkbox" checked={zoneConfig.required.includes('vest')} onChange={() => togglePPE('vest')} className="rounded border-slate-300 accent-indigo-600" />
                        <span className="text-sm text-slate-700">Vest Required</span>
                      </label>
                      <hr className="border-slate-100 my-2" />
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input type="checkbox" checked={zoneConfig.required.includes('boots')} onChange={() => togglePPE('boots')} className="rounded border-slate-300 accent-indigo-600" />
                        <span className="text-sm text-slate-700">Boots Required</span>
                      </label>
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input type="checkbox" checked={zoneConfig.required.includes('harness')} onChange={() => togglePPE('harness')} className="rounded border-slate-300 accent-indigo-600" />
                        <span className="text-sm text-slate-700">Harness Required</span>
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {(jobState === 'processing' || jobState === 'queued') && (
              <div className="glass-card p-12 text-center flex flex-col items-center justify-center animate-fade-in min-h-[400px]">
                <div className="w-16 h-16 rounded-full border-4 border-gray-100 animate-spin mb-6" style={{ borderTopColor: '#4f46e5' }}></div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Analyzing Video</h2>
                <p className="text-gray-500 mb-8 max-w-md">Processing frames through the V6-HITL model. Please wait while detections and temporal validations are performed.</p>
                <div className="w-full max-w-lg mb-6">
                  <div className="flex justify-between text-xs font-semibold text-slate-500 mb-2">
                    <span>{jobProgress?.progress || 0} / {jobProgress?.total || 0} Frames</span>
                    <span>{jobProgress && jobProgress.total > 0 ? Math.round((jobProgress.progress / jobProgress.total) * 100) : 0}%</span>
                  </div>
                  <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                    <div 
                      className="h-3 rounded-full transition-all duration-500" 
                      style={{ width: `${jobProgress && jobProgress.total > 0 ? (jobProgress.progress / jobProgress.total) * 100 : 0}%`, background: 'linear-gradient(90deg, #4f46e5, #7c3aed)' }}
                    ></div>
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-4 w-full max-w-lg mb-8">
                  <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                    <p className="text-[10px] uppercase font-bold text-slate-400">Current FPS</p>
                    <p className="text-lg font-bold text-slate-700">{jobProgress?.fps?.toFixed(1) || '0.0'}</p>
                  </div>
                  <div className="p-3 bg-purple-50 rounded-lg border border-purple-100">
                    <p className="text-[10px] uppercase font-bold text-purple-400">Workers Found</p>
                    <p className="text-lg font-bold text-purple-700">{jobProgress?.workers_detected || 0}</p>
                  </div>
                  <div className="p-3 bg-red-50 rounded-lg border border-red-100">
                    <p className="text-[10px] uppercase font-bold text-red-400">Violations</p>
                    <p className="text-lg font-bold text-red-700">{jobProgress?.violations_detected || 0}</p>
                  </div>
                </div>
                
                <button className="btn-ghost text-red-500 hover:text-red-600 hover:bg-red-50 px-4 py-2" onClick={stopProcessing}>
                  Cancel Processing
                </button>
              </div>
            )}

            {jobState === 'completed' && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in">
                {/* Video Feed */}
                <div className="lg:col-span-2 space-y-4">
                  <div className="glass-card overflow-hidden">
                    <div className="relative aspect-video bg-black flex items-center justify-center">
                      <video 
                        ref={videoRef}
                        controls 
                        className="w-full h-full object-contain"
                        src={`/results/${activeJob.id}.mp4`}
                      >
                        Your browser does not support the video tag.
                      </video>
                    </div>
                    
                    <div className="px-5 py-4 border-t border-slate-100 bg-slate-50 flex justify-between items-center">
                      <div>
                        <h3 className="font-bold text-slate-800 flex items-center gap-2">
                          <Check className="h-4 w-4 text-emerald-500" />
                          Processing Complete
                        </h3>
                        <p className="text-xs text-slate-500 mt-1">Video analyzed successfully using V6-HITL pipeline.</p>
                      </div>
                      <a href={`/results/${activeJob.id}.mp4`} download className="btn-secondary text-xs">
                        Download Video
                      </a>
                    </div>
                  </div>
                  
                  {/* Job Specific Violation Timeline */}
                  <div className="glass-card p-5">
                    <h3 className="text-sm font-bold text-slate-700 mb-4">Violation Timeline</h3>
                    {jobViolations.length === 0 ? (
                      <p className="text-sm text-slate-400">No violations detected in this video.</p>
                    ) : (
                      <div className="flex gap-3 overflow-x-auto pb-2 custom-scrollbar">
                        {jobViolations.map((v, i) => (
                          <div 
                            key={v.id} 
                            onClick={() => seekToViolation(v.video_timestamp_sec || 0)}
                            className="shrink-0 w-48 border border-slate-200 rounded-lg p-3 hover:border-blue-400 hover:shadow-sm cursor-pointer transition-all bg-white"
                          >
                            <div className="flex justify-between items-start mb-2">
                              <span className="text-xs font-bold text-slate-700">Worker #{v.worker_tracking_id}</span>
                              <span className="text-[10px] font-mono bg-slate-100 px-1.5 py-0.5 rounded text-slate-600">
                                {Math.floor((v.video_timestamp_sec || 0)/60)}:{(Math.floor(v.video_timestamp_sec || 0)%60).toString().padStart(2, '0')}
                              </span>
                            </div>
                            <p className="text-[10px] text-red-500 font-medium mb-2">Missing: {v.missing_ppe.join(', ')}</p>
                            {v.evidence_image_path && (
                              <img src={getEvidenceUrl(v.evidence_image_path)!} alt="Evidence" className="w-full h-20 object-contain rounded bg-slate-50" />
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Panel - Job Summary */}
                <div className="space-y-4">
                  <div className="glass-card p-5">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4">Job Summary</h3>
                    
                    <div className="space-y-4">
                      <div className="flex justify-between items-center pb-3 border-b border-slate-100">
                        <span className="text-sm text-slate-500">Status</span>
                        <span className="text-xs font-bold bg-emerald-100 text-emerald-700 px-2 py-1 rounded">COMPLETED</span>
                      </div>
                      <div className="flex justify-between items-center pb-3 border-b border-slate-100">
                        <span className="text-sm text-slate-500">Workers Found</span>
                        <span className="text-sm font-bold text-slate-700">{jobProgress?.workers_detected || 0}</span>
                      </div>
                      <div className="flex justify-between items-center pb-3 border-b border-slate-100">
                        <span className="text-sm text-slate-500">Violations Detected</span>
                        <span className="text-sm font-bold text-red-600">{jobProgress?.violations_detected || 0}</span>
                      </div>
                      <div className="flex justify-between items-center pb-3 border-b border-slate-100">
                        <span className="text-sm text-slate-500">Average Processing FPS</span>
                        <span className="text-sm font-bold text-slate-700">{jobProgress?.fps?.toFixed(1) || '0.0'}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-500">Total Frames</span>
                        <span className="text-sm font-bold text-slate-700">{jobProgress?.total || 0}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* ======================== MODALS ======================== */}
      {activeModal && (
        <div className="modal-backdrop" onClick={() => setActiveModal(null)}>
          <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
            
            {/* FPS Modal / Model Benchmark */}
            {activeModal === 'fps' && (
              <>
                <div className="modal-header">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: 'rgba(59, 130, 246, 0.1)' }}>
                      <Zap className="h-5 w-5 text-blue-500" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-slate-800">V6-HITL Validation Benchmark</h2>
                      <p className="text-xs text-slate-400">Model validation metrics & processing speed</p>
                    </div>
                  </div>
                  <button className="modal-close" onClick={() => setActiveModal(null)}><X className="h-4 w-4" /></button>
                </div>
                <div className="modal-body">
                  <div className="grid grid-cols-4 gap-4 mb-6">
                    <div className="p-4 rounded-xl bg-blue-50 border border-blue-100">
                      <p className="text-xs font-semibold text-blue-400 uppercase">Processing FPS</p>
                      <p className="text-2xl font-bold text-blue-600 mt-1">{displayFps}</p>
                    </div>
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                      <p className="text-xs font-semibold text-slate-400 uppercase">mAP50</p>
                      <p className="text-2xl font-bold text-slate-700 mt-1">0.84</p>
                    </div>
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                      <p className="text-xs font-semibold text-slate-400 uppercase">Helmet Recall</p>
                      <p className="text-2xl font-bold text-slate-700 mt-1">0.82</p>
                    </div>
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                      <p className="text-xs font-semibold text-slate-400 uppercase">Real-World FP</p>
                      <p className="text-2xl font-bold text-slate-700 mt-1">0</p>
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-100 p-4" style={{ height: '300px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={fpsHistory}>
                        <defs>
                          <linearGradient id="fpsGradModal" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.15} />
                            <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <XAxis dataKey="t" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                        <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
                        <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px', fontSize: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} />
                        <Area type="monotone" dataKey="fps" stroke="#3b82f6" strokeWidth={2} fill="url(#fpsGradModal)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </>
            )}

            {/* Cameras Modal */}
            {activeModal === 'cameras' && (
              <>
                <div className="modal-header">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: 'rgba(99, 102, 241, 0.1)' }}>
                      <Camera className="h-5 w-5 text-indigo-500" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-slate-800">CCTV Cameras</h2>
                      <p className="text-xs text-slate-400">Select an RTSP stream to begin real-time monitoring</p>
                    </div>
                  </div>
                  <button className="modal-close" onClick={() => setActiveModal(null)}><X className="h-4 w-4" /></button>
                </div>
                <div className="modal-body p-6">
                  
                  {/* Add Custom Camera Form */}
                  <div className="mb-6 p-4 bg-slate-50 border border-slate-200 rounded-xl">
                    <h3 className="text-sm font-semibold text-slate-700 mb-3">Add Custom Camera</h3>
                    <div className="flex gap-3">
                      <input 
                        type="text" 
                        placeholder="Camera Name (e.g. Front Gate)" 
                        className="flex-1 px-3 py-2 text-sm rounded-lg border border-slate-300 outline-none focus:border-indigo-500"
                        value={newCamName}
                        onChange={e => setNewCamName(e.target.value)}
                      />
                      <input 
                        type="text" 
                        placeholder="RTSP URL (e.g. rtsp://192.168.1.100/stream1)" 
                        className="flex-[2] px-3 py-2 text-sm rounded-lg border border-slate-300 outline-none focus:border-indigo-500"
                        value={newCamUrl}
                        onChange={e => setNewCamUrl(e.target.value)}
                      />
                      <button 
                        className="btn-primary px-4 py-2 text-sm shrink-0"
                        disabled={!newCamName || !newCamUrl}
                        onClick={async () => {
                          if (!newCamName || !newCamUrl) return;
                          await fetch('/api/v1/cameras', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                              name: newCamName,
                              source_type: "RTSP",
                              rtsp_url: newCamUrl,
                              is_active: true
                            })
                          });
                          const res = await fetch('/api/v1/cameras');
                          if(res.ok) setCameras(await res.json());
                          setNewCamName('');
                          setNewCamUrl('');
                        }}
                      >
                        Add Camera
                      </button>
                    </div>
                  </div>

                  {cameras.length === 0 ? (
                    <div className="py-12 text-center text-sm text-slate-400">
                      No cameras configured yet.
                      <div className="mt-4">
                        <button 
                          className="btn-primary inline-block text-xs px-4"
                          onClick={async () => {
                            await fetch('/api/v1/cameras', {
                              method: 'POST',
                              headers: {'Content-Type': 'application/json'},
                              body: JSON.stringify({
                                name: "Demo RTSP Camera",
                                source_type: "RTSP",
                                rtsp_url: "rtsp://demo-cam:8554/live",
                                is_active: true
                              })
                            });
                            const res = await fetch('/api/v1/cameras');
                            if(res.ok) setCameras(await res.json());
                          }}
                        >
                          + Add Demo RTSP Camera
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {cameras.map((cam) => (
                        <div key={cam.id} className="border border-slate-200 rounded-xl overflow-hidden hover:border-indigo-300 transition-colors bg-white shadow-sm flex flex-col">
                          <div className="bg-slate-100 aspect-video relative flex items-center justify-center">
                            <img 
                              src={`/api/v1/cameras/${cam.id}/snapshot`}
                              alt={cam.name}
                              className="absolute inset-0 w-full h-full object-cover"
                              onError={(e) => {
                                (e.target as HTMLImageElement).style.display = 'none';
                              }}
                            />
                            <div className="absolute inset-0 bg-black/10 flex items-center justify-center">
                              <Video className="h-8 w-8 text-white/50" />
                            </div>
                            {cam.is_active && (
                              <div className="absolute top-2 left-2 bg-emerald-500 text-white text-[10px] font-bold px-2 py-0.5 rounded shadow-sm flex items-center gap-1">
                                <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse"></div> LIVE
                              </div>
                            )}
                          </div>
                          <div className="p-4 flex-1 flex flex-col justify-between">
                            <div className="flex items-start justify-between gap-2">
                              <div className="min-w-0 flex-1">
                                <h3 className="font-bold text-slate-800 mb-1 truncate">{cam.name}</h3>
                                <p className="text-[10px] text-slate-500 font-mono truncate">{cam.rtsp_url || cam.local_video_path}</p>
                              </div>
                              <button
                                className="p-1.5 shrink-0 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                                title="Delete Camera"
                                onClick={async (e) => {
                                  e.stopPropagation();
                                  await fetch(`/api/v1/cameras/${cam.id}`, { method: 'DELETE' });
                                  const res = await fetch('/api/v1/cameras');
                                  if(res.ok) setCameras(await res.json());
                                }}
                              >
                                <Trash2 className="h-4 w-4" />
                              </button>
                            </div>
                            <button 
                              className="mt-4 w-full py-2 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 font-semibold text-xs rounded-lg transition-colors flex items-center justify-center gap-2"
                              onClick={() => {
                                setUploadStatus('CAMERA_SELECTED');
                                setUploadedVideo({ id: cam.id, filename: cam.name, filepath: cam.rtsp_url || cam.local_video_path, isCamera: true });
                                setActiveJob(null);
                                setActiveModal(null);
                              }}
                            >
                              <Play className="h-3 w-3" /> Connect & Process
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}

            {/* Violations Modal */}
            {activeModal === 'violations' && (
              <>
                <div className="modal-header">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: 'rgba(239, 68, 68, 0.1)' }}>
                      <AlertTriangle className="h-5 w-5 text-red-500" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-slate-800">All Violations</h2>
                      <p className="text-xs text-slate-400">{allViolations.length} total violations detected</p>
                    </div>
                  </div>
                  <button className="modal-close" onClick={() => setActiveModal(null)}><X className="h-4 w-4" /></button>
                </div>
                <div className="modal-body">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {allViolations.length === 0 ? (
                      <div className="col-span-2 py-12 text-center text-sm text-slate-400">No violations recorded yet.</div>
                    ) : allViolations.map((event) => (
                      <div key={event.id} className="evidence-card">
                        {event.evidence_image_path ? (
                          <img 
                            src={getEvidenceUrl(event.evidence_image_path)!}
                            alt={`Worker #${event.worker_tracking_id}`}
                          />
                        ) : (
                          <div className="w-full h-[180px] bg-slate-100 flex items-center justify-center text-slate-300">
                            <Eye className="h-8 w-8" />
                          </div>
                        )}
                        <div className="info">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm font-semibold text-slate-700">Worker #{event.worker_tracking_id}</span>
                            {event.acknowledged ? (
                              <span className="badge badge-success text-[10px] py-0.5 px-1.5">Resolved</span>
                            ) : (
                              <span className="badge badge-danger text-[10px] py-0.5 px-1.5">Active</span>
                            )}
                          </div>
                          <p className="text-xs text-red-500 mb-1">Missing: {event.missing_ppe.join(', ')}</p>
                          <p className="text-[10px] text-slate-400">{new Date(event.timestamp).toLocaleString()}</p>
                          
                          {event.feedback_correct !== null && (
                            <div className="mt-2 text-[10px] px-2 py-1 bg-blue-50 text-blue-600 rounded border border-blue-100 font-medium text-center">
                              Feedback Provided ✓
                            </div>
                          )}
                          
                          {!event.acknowledged && (
                            <button 
                              onClick={() => acknowledgeViolation(event.id)}
                              className="mt-2 w-full text-xs py-1.5 bg-slate-50 rounded-lg border border-slate-200 hover:bg-slate-100 text-slate-600 flex items-center justify-center gap-1 font-medium"
                            >
                              <Check className="h-3 w-3" /> Acknowledge
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            {/* Training Modal */}
            {activeModal === 'training' && (
              <>
                <div className="modal-header">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: 'rgba(59, 130, 246, 0.1)' }}>
                      <Zap className="h-5 w-5 text-blue-500" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-slate-800">Model Training (HITL)</h2>
                      <p className="text-xs text-slate-400">Review detections to improve model accuracy</p>
                    </div>
                  </div>
                  <button className="modal-close" onClick={() => setActiveModal(null)}><X className="h-4 w-4" /></button>
                </div>
                <div className="modal-body max-w-3xl mx-auto w-full p-6">
                  {trainingQueue.length === 0 ? (
                    <div className="py-20 flex flex-col items-center justify-center text-center">
                      <ShieldCheck className="h-16 w-16 text-emerald-400 mb-4 opacity-50" />
                      <h3 className="text-xl font-bold text-slate-700 mb-2">All caught up!</h3>
                      <p className="text-slate-500 text-sm max-w-md">There are no pending detections to review. Your feedback helps improve the model for future inference.</p>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      {/* Image Side */}
                      <div className="flex flex-col">
                        <div className="rounded-xl overflow-hidden border border-slate-200 bg-white shadow-sm mb-3">
                          {trainingQueue[currentTrainingIndex].evidence_image_path ? (
                            <img 
                              src={getEvidenceUrl(trainingQueue[currentTrainingIndex].evidence_image_path)!}
                              alt="Training Evidence"
                              className="w-full h-auto object-contain bg-slate-50 max-h-[400px]"
                            />
                          ) : (
                            <div className="w-full h-[300px] bg-slate-100 flex items-center justify-center text-slate-300">
                              <Eye className="h-12 w-12" />
                            </div>
                          )}
                        </div>
                        <div className="flex justify-between items-center px-1">
                          <span className="text-xs font-semibold text-slate-500">
                            {currentTrainingIndex + 1} of {trainingQueue.length}
                          </span>
                          <span className="text-xs font-medium bg-red-50 text-red-600 px-2 py-1 rounded border border-red-100">
                            Prediction: Missing {trainingQueue[currentTrainingIndex].missing_ppe.join(', ')}
                          </span>
                        </div>
                      </div>
                      
                      {/* Feedback Side */}
                      <div className="flex flex-col">
                        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm flex-1">
                          <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wide mb-6">Ground Truth Labels</h3>
                          
                          <div className="space-y-6">
                            {/* Correctness */}
                            <div>
                              <p className="text-sm font-medium text-slate-700 mb-3">Was the model's detection correct?</p>
                              <div className="flex gap-3">
                                <button 
                                  onClick={() => setFeedback({...feedback, correct: true})}
                                  className={`flex-1 py-2.5 rounded-lg border text-sm font-medium transition-all ${feedback.correct ? 'bg-emerald-50 border-emerald-500 text-emerald-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                                >
                                  👍 Yes
                                </button>
                                <button 
                                  onClick={() => setFeedback({...feedback, correct: false})}
                                  className={`flex-1 py-2.5 rounded-lg border text-sm font-medium transition-all ${!feedback.correct ? 'bg-red-50 border-red-500 text-red-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                                >
                                  👎 No
                                </button>
                              </div>
                            </div>
                            
                            <hr className="border-slate-100" />
                            
                            {/* Labels */}
                            <div>
                              <p className="text-sm font-medium text-slate-700 mb-3">Does the person have a HELMET?</p>
                              <div className="flex gap-3">
                                <button 
                                  onClick={() => setFeedback({...feedback, helmet: true})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${feedback.helmet ? 'bg-blue-50 border-blue-500 text-blue-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                                >
                                  Yes
                                </button>
                                <button 
                                  onClick={() => setFeedback({...feedback, helmet: false})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${!feedback.helmet ? 'bg-slate-100 border-slate-300 text-slate-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                                >
                                  No
                                </button>
                              </div>
                            </div>
                            
                            <div>
                              <p className="text-sm font-medium text-slate-700 mb-3">Does the person have a VEST?</p>
                              <div className="flex gap-3">
                                <button 
                                  onClick={() => setFeedback({...feedback, vest: true})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${feedback.vest ? 'bg-blue-50 border-blue-500 text-blue-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                                >
                                  Yes
                                </button>
                                <button 
                                  onClick={() => setFeedback({...feedback, vest: false})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${!feedback.vest ? 'bg-slate-100 border-slate-300 text-slate-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                                >
                                  No
                                </button>
                              </div>
                            </div>

                            <div>
                              <p className="text-sm font-medium text-slate-700 mb-3 mt-4">Does the person have BOOTS?</p>
                              <div className="flex gap-3">
                                <button 
                                  onClick={() => setFeedback({...feedback, boots: true})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${feedback.boots ? 'bg-blue-50 border-blue-500 text-blue-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                                >
                                  Yes
                                </button>
                                <button 
                                  onClick={() => setFeedback({...feedback, boots: false})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${!feedback.boots ? 'bg-slate-100 border-slate-300 text-slate-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                                >
                                  No
                                </button>
                              </div>
                            </div>
                          </div>
                          
                          <div className="mt-8 pt-4 border-t border-slate-100">
                            <button 
                              onClick={() => submitTrainingFeedback(trainingQueue[currentTrainingIndex].id)}
                              className="w-full py-3 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-bold text-sm shadow-sm transition-all"
                            >
                              Submit Feedback & Next
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}

            {/* Resolution Rate Modal */}
            {activeModal === 'resolution' && (
              <>
                <div className="modal-header">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: 'rgba(16, 185, 129, 0.1)' }}>
                      <ShieldCheck className="h-5 w-5 text-emerald-500" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-slate-800">Compliance Overview</h2>
                      <p className="text-xs text-slate-400">Acknowledged vs Active violations</p>
                    </div>
                  </div>
                  <button className="modal-close" onClick={() => setActiveModal(null)}><X className="h-4 w-4" /></button>
                </div>
                <div className="modal-body">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="flex flex-col items-center justify-center">
                      <div className="relative" style={{ width: 200, height: 200 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={donutData}
                              cx="50%"
                              cy="50%"
                              innerRadius={60}
                              outerRadius={90}
                              paddingAngle={4}
                              dataKey="value"
                            >
                              {donutData.map((entry, index) => (
                                <Cell key={index} fill={entry.color} />
                              ))}
                            </Pie>
                          </PieChart>
                        </ResponsiveContainer>
                        <div className="donut-center">
                          <p className="text-3xl font-bold text-slate-800">{resolutionRate}%</p>
                          <p className="text-[10px] text-slate-400 uppercase font-semibold">Resolved</p>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-100">
                        <p className="text-xs font-semibold text-emerald-500 uppercase">Resolved</p>
                        <p className="text-2xl font-bold text-emerald-600 mt-1">{stats?.acknowledged || 0}</p>
                      </div>
                      <div className="p-4 rounded-xl bg-red-50 border border-red-100">
                        <p className="text-xs font-semibold text-red-400 uppercase">Active</p>
                        <p className="text-2xl font-bold text-red-500 mt-1">{stats?.active_violations || 0}</p>
                      </div>
                      <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                        <p className="text-xs font-semibold text-slate-400 uppercase">Total</p>
                        <p className="text-2xl font-bold text-slate-700 mt-1">{stats?.total_violations || 0}</p>
                      </div>
                    </div>
                  </div>

                  {/* Show violation evidence in this modal too */}
                  <div className="mt-6 border-t border-slate-100 pt-4">
                    <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Recent Resolved Violations</p>
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                      {allViolations.filter(v => v.acknowledged).slice(0, 8).map((event) => (
                        <div key={event.id} className="evidence-card">
                          {event.evidence_image_path ? (
                            <img src={getEvidenceUrl(event.evidence_image_path)!} alt={`Worker #${event.worker_tracking_id}`} style={{ height: 120 }} />
                          ) : (
                            <div className="w-full bg-slate-100 flex items-center justify-center text-slate-300" style={{ height: 120 }}><Eye className="h-6 w-6" /></div>
                          )}
                          <div className="info py-2 px-2">
                            <p className="text-xs font-medium text-slate-600">Worker #{event.worker_tracking_id}</p>
                            <p className="text-[10px] text-slate-400">{event.missing_ppe.join(', ')}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* Workers Modal */}
            {activeModal === 'workers' && (
              <>
                <div className="modal-header">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
                      <Users className="h-5 w-5 text-purple-500" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-slate-800">Worker Directory</h2>
                      <p className="text-xs text-slate-400">{Object.keys(workerData).length} unique workers tracked</p>
                    </div>
                  </div>
                  <button className="modal-close" onClick={() => setActiveModal(null)}><X className="h-4 w-4" /></button>
                </div>
                <div className="modal-body">
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.values(workerData).map((worker: any) => {
                      const latestViolation = worker.violations[0];
                      const unresolved = worker.violations.filter((v: any) => !v.acknowledged).length;
                      return (
                        <div key={worker.id} className="rounded-xl border border-slate-100 overflow-hidden bg-white hover:shadow-md transition-shadow">
                          {/* Show latest evidence image for this worker */}
                          {latestViolation?.evidence_image_path ? (
                            <img 
                              src={getEvidenceUrl(latestViolation.evidence_image_path)!}
                              alt={`Worker #${worker.id}`}
                              className="w-full h-36 object-contain bg-slate-50"
                            />
                          ) : (
                            <div className="w-full h-36 bg-slate-50 flex items-center justify-center">
                              <Users className="h-10 w-10 text-slate-200" />
                            </div>
                          )}
                          <div className="p-3">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm font-bold text-slate-700">Worker #{worker.id}</span>
                              {unresolved > 0 ? (
                                <span className="badge badge-danger text-[10px] py-0.5 px-1.5">{unresolved} Active</span>
                              ) : (
                                <span className="badge badge-success text-[10px] py-0.5 px-1.5">Compliant</span>
                              )}
                            </div>
                            <p className="text-xs text-slate-400">{worker.count} total violation{worker.count !== 1 ? 's' : ''}</p>
                            <div className="mt-2 flex flex-wrap gap-1">
                              {[...new Set(worker.violations.flatMap((v: any) => v.missing_ppe))].map((ppe: any) => (
                                <span key={ppe} className="text-[10px] px-1.5 py-0.5 bg-red-50 text-red-500 rounded border border-red-100">{ppe}</span>
                              ))}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                    {Object.keys(workerData).length === 0 && (
                      <div className="col-span-3 py-12 text-center text-sm text-slate-400">No workers tracked yet.</div>
                    )}
                  </div>
                </div>
              </>
            )}

            {/* Cameras Modal */}
            {activeModal === 'cameras' && (
              <>
                <div className="modal-header">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center bg-slate-100">
                      <Camera className="h-5 w-5 text-slate-600" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-slate-800">Video Sources</h2>
                      <p className="text-xs text-slate-400">View uploaded video files</p>
                    </div>
                  </div>
                  <button className="modal-close" onClick={() => setActiveModal(null)}><X className="h-4 w-4" /></button>
                </div>
                <div className="modal-body p-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {cameras.length === 0 ? (
                      <div className="col-span-2 py-10 text-center border-2 border-dashed border-slate-200 rounded-xl">
                        <Video className="h-10 w-10 text-slate-300 mx-auto mb-2" />
                        <p className="text-sm text-slate-500 font-medium">No cameras connected</p>
                        <button className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg text-xs font-semibold hover:bg-blue-600">
                          + Add IP Camera
                        </button>
                      </div>
                    ) : cameras.map(cam => (
                      <div key={cam.id} className="p-4 rounded-xl border border-slate-200 bg-white shadow-sm flex items-start gap-4">
                        <div className="w-24 h-16 bg-slate-100 rounded overflow-hidden flex-shrink-0 flex items-center justify-center">
                          <Video className="h-6 w-6 text-slate-300" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-sm text-slate-800">{cam.name}</h4>
                          <p className="text-xs text-slate-400 truncate max-w-[150px]" title={cam.url}>{cam.url}</p>
                          <div className="mt-2 flex gap-2">
                            <span className={`text-[10px] px-2 py-0.5 rounded font-medium ${cam.is_active ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-100 text-slate-500'}`}>
                              {cam.is_active ? 'Online' : 'Offline'}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            {/* Reports Modal */}
            {activeModal === 'reports' && (
              <>
                <div className="modal-header">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center bg-blue-50">
                      <FileText className="h-5 w-5 text-blue-500" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-slate-800">Safety Reports</h2>
                      <p className="text-xs text-slate-400">Daily, weekly and monthly compliance</p>
                    </div>
                  </div>
                  <button className="modal-close" onClick={() => setActiveModal(null)}><X className="h-4 w-4" /></button>
                </div>
                <div className="modal-body p-6">
                  <div className="flex gap-2 mb-6">
                    <button className="px-3 py-1 bg-blue-500 text-white rounded text-xs font-semibold">Daily</button>
                    <button className="px-3 py-1 bg-slate-100 text-slate-600 rounded text-xs font-medium hover:bg-slate-200">Weekly</button>
                    <button className="px-3 py-1 bg-slate-100 text-slate-600 rounded text-xs font-medium hover:bg-slate-200">Monthly</button>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="p-4 bg-slate-50 border border-slate-100 rounded-xl">
                      <p className="text-xs text-slate-500 mb-1">Total Workers</p>
                      <p className="text-2xl font-bold text-slate-800">{stats?.unique_workers || 0}</p>
                    </div>
                    <div className="p-4 bg-slate-50 border border-slate-100 rounded-xl">
                      <p className="text-xs text-slate-500 mb-1">Violations Logged</p>
                      <p className="text-2xl font-bold text-slate-800">{stats?.active_violations || 0}</p>
                    </div>
                    <div className="p-4 bg-emerald-50 border border-emerald-100 rounded-xl">
                      <p className="text-xs text-emerald-600 mb-1">Compliance Rate</p>
                      <p className="text-2xl font-bold text-emerald-700">{resolutionRate}%</p>
                    </div>
                  </div>
                  
                  <div className="h-[200px] w-full border border-slate-200 rounded-xl bg-white flex items-center justify-center flex-col">
                    <AreaChart className="opacity-50" />
                    <p className="text-sm text-slate-400 mt-2 font-medium">Compliance Trend (Mock Data)</p>
                  </div>
                  
                  <div className="mt-6 flex justify-end">
                    <button className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-bold shadow hover:bg-blue-700">Download PDF Report</button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
