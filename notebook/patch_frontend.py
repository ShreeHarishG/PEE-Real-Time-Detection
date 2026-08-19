import os

def patch_frontend(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update feedback state
    content = content.replace(
        "const [feedback, setFeedback] = useState({ correct: true, helmet: false, vest: false, boots: false });",
        "const [feedback, setFeedback] = useState({ correct: true, helmet: false, vest: false, boots: false, harness: false });"
    )
    content = content.replace(
        "setFeedback({ correct: true, helmet: false, vest: false, boots: false });",
        "setFeedback({ correct: true, helmet: false, vest: false, boots: false, harness: false });"
    )
    
    # Update feedback submission
    content = content.replace(
        "feedback_boots: feedback.boots\n",
        "feedback_boots: feedback.boots,\n          feedback_harness: feedback.harness\n"
    )
    
    # Update text
    content = content.replace("V4-Boots", "V5-Harness")
    
    # Update Harness Checkbox
    if 'disabled' in content and 'Harness Required' in content:
        # For page.tsx
        content = content.replace(
            '<label className="flex items-center gap-3 opacity-50 cursor-not-allowed">\n                        <input type="checkbox" disabled className="rounded border-slate-300" />\n                        <span className="text-sm text-slate-400">Harness Required <span className="text-xs text-red-400 ml-1">(UNSUPPORTED)</span></span>\n                      </label>',
            '<label className="flex items-center gap-3 cursor-pointer">\n                        <input type="checkbox" checked={zoneConfig.required.includes(\'harness\')} onChange={() => togglePPE(\'harness\')} className="rounded border-slate-300 accent-indigo-600" />\n                        <span className="text-sm text-slate-700">Harness Required</span>\n                      </label>'
        )
        # For live/page.tsx (dark mode styling)
        content = content.replace(
            '<label className="flex items-center gap-3 opacity-50 cursor-not-allowed">\n                        <input type="checkbox" disabled className="rounded border-slate-700 bg-slate-800" />\n                        <span className="text-sm text-slate-400">Harness Required <span className="text-xs text-red-400 ml-1">(UNSUPPORTED)</span></span>\n                      </label>',
            '<label className="flex items-center gap-3 cursor-pointer">\n                        <input type="checkbox" checked={zoneConfig.required.includes(\'harness\')} onChange={() => togglePPE(\'harness\')} className="rounded border-slate-700 bg-slate-800 accent-emerald-500" />\n                        <span className="text-sm text-slate-200">Harness Required</span>\n                      </label>'
        )

    # Add Harness Feedback UI
    # In page.tsx (light theme):
    boots_ui_light = """<p className="text-sm font-medium text-slate-700 mb-3 mt-4">Does the person have BOOTS?</p>
                              <div className="flex gap-3">
                                <button 
                                  onClick={() => setFeedback({...feedback, boots: true})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${feedback.boots ? 'bg-blue-50 border-blue-500 text-blue-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                                >
                                  Yes, they do
                                </button>
                                <button 
                                  onClick={() => setFeedback({...feedback, boots: false})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${!feedback.boots ? 'bg-slate-100 border-slate-300 text-slate-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                                >
                                  No, they don't
                                </button>
                              </div>"""
                              
    harness_ui_light = """<p className="text-sm font-medium text-slate-700 mb-3 mt-4">Does the person have a HARNESS?</p>
                              <div className="flex gap-3">
                                <button 
                                  onClick={() => setFeedback({...feedback, harness: true})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${feedback.harness ? 'bg-blue-50 border-blue-500 text-blue-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                                >
                                  Yes, they do
                                </button>
                                <button 
                                  onClick={() => setFeedback({...feedback, harness: false})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${!feedback.harness ? 'bg-slate-100 border-slate-300 text-slate-700' : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}
                                >
                                  No, they don't
                                </button>
                              </div>"""
                              
    if boots_ui_light in content and harness_ui_light not in content:
        content = content.replace(boots_ui_light, boots_ui_light + "\n" + harness_ui_light)

    # In live/page.tsx (dark theme):
    boots_ui_dark = """<p className="text-sm font-medium text-slate-200 mb-3 mt-4">Does the person have BOOTS?</p>
                              <div className="flex gap-3">
                                <button 
                                  onClick={() => setFeedback({...feedback, boots: true})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${feedback.boots ? 'bg-blue-50 border-blue-500 text-blue-700' : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'}`}
                                >
                                  Yes, they do
                                </button>
                                <button 
                                  onClick={() => setFeedback({...feedback, boots: false})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${!feedback.boots ? 'bg-slate-900 border-slate-600 text-slate-200' : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'}`}
                                >
                                  No, they don't
                                </button>
                              </div>"""
                              
    harness_ui_dark = """<p className="text-sm font-medium text-slate-200 mb-3 mt-4">Does the person have a HARNESS?</p>
                              <div className="flex gap-3">
                                <button 
                                  onClick={() => setFeedback({...feedback, harness: true})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${feedback.harness ? 'bg-blue-50 border-blue-500 text-blue-700' : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'}`}
                                >
                                  Yes, they do
                                </button>
                                <button 
                                  onClick={() => setFeedback({...feedback, harness: false})}
                                  className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all ${!feedback.harness ? 'bg-slate-900 border-slate-600 text-slate-200' : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'}`}
                                >
                                  No, they don't
                                </button>
                              </div>"""
                              
    if boots_ui_dark in content and harness_ui_dark not in content:
        content = content.replace(boots_ui_dark, boots_ui_dark + "\n" + harness_ui_dark)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

patch_frontend('frontend/src/app/page.tsx')
patch_frontend('frontend/src/app/live/page.tsx')
# Also update sidebar
with open('frontend/src/components/Sidebar.tsx', 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace("V4-Boots", "V5-Harness")
with open('frontend/src/components/Sidebar.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched frontend pages")
