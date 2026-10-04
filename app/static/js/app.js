/**
 * RadAssist AI - Interactive Clinical Workstation & DICOM Viewer Client
 */

// Global State
const state = {
  currentTab: 'worklist',
  studies: [],
  selectedStudyId: null,
  currentStudy: null,
  activeModelId: 'efficientnet_b0_cxr',
  // Viewer state
  zoom: 1.0,
  panX: 0,
  panY: 0,
  isPanning: false,
  startMouseX: 0,
  startMouseY: 0,
  isInverted: false,
  // Layer toggles
  layers: {
    gradcam: true,
    bbox: true,
    segmentation: true
  },
  gradcamAlpha: 0.55,
  cachedImages: {
    base: null,
    gradcam: null,
    bbox: null,
    segmentation: null
  }
};

// DOM Ready Initialization
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initWorklistActions();
  initViewerControls();
  initCanvasInteraction();
  loadWorklist();
  loadModels();
  initUploadForm();
  initReviewForm();
  initMLOps();
  initDataQuality();
});

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  })[character]);
}

function initWorklistActions() {
  document.getElementById('worklistTableBody')?.addEventListener('click', event => {
    const action = event.target instanceof Element
      ? event.target.closest('[data-study-id]')
      : null;
    if (action) openStudyInViewer(action.dataset.studyId);
  });
}

// ==========================================
// 1. Navigation & Tab Management
// ==========================================
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab-btn');
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      switchTab(target);
    });
  });
}

function switchTab(tabId) {
  state.currentTab = tabId;
  document.querySelectorAll('.nav-tab-btn').forEach(b => {
    if (b.dataset.tab === tabId) {
      b.classList.add('border-cyan-400', 'text-cyan-400');
      b.classList.remove('border-transparent', 'text-slate-400');
      b.setAttribute('aria-current', 'page');
    } else {
      b.classList.remove('border-cyan-400', 'text-cyan-400');
      b.classList.add('border-transparent', 'text-slate-400');
      b.removeAttribute('aria-current');
    }
  });

  document.querySelectorAll('.tab-view-panel').forEach(panel => {
    if (panel.id === `tab-${tabId}`) {
      panel.classList.remove('hidden');
    } else {
      panel.classList.add('hidden');
    }
  });

  if (tabId === 'worklist') loadWorklist();
  if (tabId === 'mlops') refreshMLOps();
}

// ==========================================
// 2. Worklist & Clinical Queue Operations
// ==========================================
async function loadWorklist() {
  const priorityFilter = document.getElementById('filterPriority')?.value || 'ALL';
  const statusFilter = document.getElementById('filterStatus')?.value || 'ALL';

  try {
    const [res, allRes] = await Promise.all([
      fetch(`/studies?priority=${priorityFilter}&status=${statusFilter}`),
      fetch('/studies?priority=ALL&status=ALL')
    ]);
    if (!res.ok || !allRes.ok) {
      const failed = !res.ok ? res : allRes;
      throw new Error(`Worklist request failed (${failed.status}).`);
    }
    const [data, allStudies] = await Promise.all([res.json(), allRes.json()]);
    state.studies = data;
    renderWorklistTable(data);
    updateWorklistStats(allStudies);
  } catch (err) {
    console.error('Failed to load worklist:', err);
    const tbody = document.getElementById('worklistTableBody');
    if (tbody) {
      const cell = document.createElement('td');
      cell.colSpan = 8;
      cell.className = 'text-center py-8 text-red-700';
      cell.textContent = `Could not load studies: ${err.message}`;
      const row = document.createElement('tr');
      row.append(cell);
      tbody.replaceChildren(row);
    }
  }
}

function updateWorklistStats(studies) {
  const statCount = studies.filter(s => s.triage?.triage_priority === 'STAT').length;
  const urgentCount = studies.filter(s => s.triage?.triage_priority === 'URGENT').length;
  const routineCount = studies.filter(s => s.triage?.triage_priority === 'ROUTINE').length;
  const statBadge = document.getElementById('statCountBadge');
  if (statBadge) statBadge.textContent = `${statCount} STAT`;
  const urgentBadge = document.getElementById('urgentCountBadge');
  if (urgentBadge) urgentBadge.textContent = `${urgentCount} Urgent`;
  const routineBadge = document.getElementById('routineCountBadge');
  if (routineBadge) routineBadge.textContent = `${routineCount} Routine`;
}

function renderWorklistTable(studies) {
  const tbody = document.getElementById('worklistTableBody');
  if (!tbody) return;

  if (studies.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center py-8 text-slate-400">No studies match the selected filters.</td></tr>`;
    return;
  }

  tbody.innerHTML = studies.map(s => {
    const priority = s.triage?.triage_priority || 'ROUTINE';
    const badgeClass = priority === 'STAT' ? 'badge-stat' : (priority === 'URGENT' ? 'badge-urgent' : 'badge-routine');
    const topFinding = s.ai_result?.predictions?.[0]?.finding || 'Processing...';
    const probability = s.ai_result?.predictions?.[0]?.probability;
    const topProb = Number.isFinite(probability) ? `${(probability * 100).toFixed(0)}%` : '--';
    const riskScore = Number(s.triage?.risk_score || 0).toFixed(2);
    const status = s.status || 'PENDING_REVIEW';
    const statusColor = status === 'SIGNED_OFF' ? 'text-emerald-400' : (status === 'OVERRIDDEN' ? 'text-amber-400' : 'text-cyan-400');

    return `
      <tr class="cursor-pointer hover:bg-slate-800/60 transition" data-study-id="${escapeHtml(s.study_id)}">
        <td class="font-mono text-cyan-300 font-semibold">${escapeHtml(s.study_id)}</td>
        <td>
          <span class="clinical-badge ${badgeClass}">
            ${priority === 'STAT' ? '⚡ ' : ''}${priority}
          </span>
        </td>
        <td>
          <div class="font-medium text-slate-200">${escapeHtml(s.patient_info?.patient_name || 'ANONYMOUS')}</div>
          <div class="text-xs text-slate-400 font-mono">${escapeHtml(s.patient_info?.patient_id || 'PT-UNKNOWN')} (${escapeHtml(s.patient_info?.patient_age || '--')}, ${escapeHtml(s.patient_info?.patient_sex || '--')})</div>
        </td>
        <td>
          <span class="px-2 py-0.5 rounded text-xs bg-slate-700 text-slate-300 font-mono font-bold">${escapeHtml(s.modality || 'CR')}</span>
          <span class="text-xs text-slate-400 ml-1">${escapeHtml(s.body_part || 'CHEST')}</span>
        </td>
        <td>
          <div class="text-slate-100 font-medium">${escapeHtml(topFinding)}</div>
          <div class="text-xs text-slate-400">Calibrated Conf: <span class="text-cyan-400 font-mono">${topProb}</span></div>
        </td>
        <td>
          <div class="flex items-center gap-2">
            <span class="font-mono font-semibold ${priority === 'STAT' ? 'text-red-400' : 'text-slate-300'}">${riskScore}</span>
            <div class="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div class="h-full ${priority === 'STAT' ? 'bg-red-500' : (priority === 'URGENT' ? 'bg-amber-500' : 'bg-emerald-500')}" style="width: ${Math.min(100, riskScore * 100)}%"></div>
            </div>
          </div>
        </td>
        <td>
          <span class="text-xs font-semibold ${statusColor}">${escapeHtml(status.replace('_', ' '))}</span>
        </td>
        <td>
          <button class="tool-btn text-xs py-1 px-3" type="button" data-study-id="${escapeHtml(s.study_id)}">
            View / Review ➔
          </button>
        </td>
      </tr>
    `;
  }).join('');
}

// ==========================================
// 3. Medical Viewer & Layer Compositor
// ==========================================
async function openStudyInViewer(studyId) {
  state.selectedStudyId = studyId;
  switchTab('viewer');
  window.scrollTo({ top: 0, behavior: 'auto' });

  try {
    const res = await fetch(`/studies/${studyId}`);
    if (!res.ok) throw new Error(`Study request failed (${res.status}).`);
    const study = await res.json();
    state.currentStudy = study;
    renderViewerHUD(study);
    renderClinicalAssessment(study);
    await loadStudyImages(study);
    renderCanvas();
  } catch (err) {
    console.error('Error loading study into viewer:', err);
    alert(`Could not open study: ${err.message}`);
  }
}

async function loadStudyImages(study) {
  // Reset cached image objects
  state.cachedImages = { base: null, gradcam: null, bbox: null, segmentation: null };

  const loadImg = (src) => new Promise(resolve => {
    if (!src) return resolve(null);
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });

  const baseSrc = study.image_data_base64;
  const gradcamSrc = study.ai_result?.explainability?.gradcam_overlay_base64;
  const bboxSrc = study.ai_result?.explainability?.bbox_overlay_base64;
  const segSrc = study.ai_result?.segmentation?.mask_overlay_base64;

  const [baseImg, gcImg, bbImg, segImg] = await Promise.all([
    loadImg(baseSrc),
    loadImg(gradcamSrc),
    loadImg(bboxSrc),
    loadImg(segSrc)
  ]);
  if (!baseImg) throw new Error('Study image preview is missing or could not be decoded.');

  state.cachedImages.base = baseImg;
  state.cachedImages.gradcam = gcImg;
  state.cachedImages.bbox = bbImg;
  state.cachedImages.segmentation = segImg;
}

function renderCanvas() {
  const canvas = document.getElementById('medicalCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  if (!state.cachedImages.base) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const baseImg = state.cachedImages.base;
  canvas.width = baseImg.naturalWidth || 512;
  canvas.height = baseImg.naturalHeight || 512;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();

  // Apply Pan and Zoom
  ctx.translate(canvas.width / 2 + state.panX, canvas.height / 2 + state.panY);
  ctx.scale(state.zoom, state.zoom);
  ctx.translate(-canvas.width / 2, -canvas.height / 2);

  // 1. Draw Base Grayscale Radiograph
  if (state.isInverted) {
    ctx.filter = 'invert(100%)';
  } else {
    ctx.filter = 'none';
  }
  ctx.drawImage(baseImg, 0, 0, canvas.width, canvas.height);
  ctx.filter = 'none'; // reset filter for overlays

  // 2. Draw Grad-CAM Heatmap Layer
  if (state.layers.gradcam && state.cachedImages.gradcam) {
    ctx.globalAlpha = state.gradcamAlpha;
    ctx.drawImage(state.cachedImages.gradcam, 0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 1.0;
  }

  // 3. Draw MONAI / U-Net Segmentation Mask
  if (state.layers.segmentation && state.cachedImages.segmentation) {
    ctx.globalAlpha = 0.65;
    ctx.drawImage(state.cachedImages.segmentation, 0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 1.0;
  }

  // 4. Draw Bounding Box Localization
  if (state.layers.bbox && state.cachedImages.bbox) {
    ctx.drawImage(state.cachedImages.bbox, 0, 0, canvas.width, canvas.height);
  }

  ctx.restore();
}

function renderViewerHUD(study) {
  const p = study.patient_info || {};
  const metaLeft = document.getElementById('hudTopLeft');
  if (metaLeft) {
    metaLeft.innerHTML = `
      <div><strong>${escapeHtml(p.patient_name || 'ANONYMOUS')}</strong></div>
      <div>ID: ${escapeHtml(p.patient_id || 'PT-UNKNOWN')} | ${escapeHtml(p.patient_age || '--')} / ${escapeHtml(p.patient_sex || '--')}</div>
      <div>ACC: ${escapeHtml(p.accession_number || '--')}</div>
      <div>MOD: ${escapeHtml(study.modality || 'CR')} | ${escapeHtml(study.body_part || 'CHEST')}</div>
    `;
  }

  const metaRight = document.getElementById('hudTopRight');
  if (metaRight) {
    const priority = study.triage?.triage_priority || 'ROUTINE';
    metaRight.innerHTML = `
      <div class="font-bold ${priority === 'STAT' ? 'text-red-400' : 'text-amber-400'}">PRIORITY: ${escapeHtml(priority)}</div>
      <div>STUDY: ${escapeHtml(study.study_id)}</div>
      <div>MATRIX: ${study.dimensions?.[1] || 512} x ${study.dimensions?.[0] || 512}</div>
      <div>SPACING: 0.143 mm</div>
    `;
  }
}

function renderClinicalAssessment(study) {
  const ai = study.ai_result || {};
  const triage = study.triage || {};
  const topPred = ai.predictions?.[0] || {};

  // Finding Badge
  const findingTitle = document.getElementById('viewerPrimaryFinding');
  if (findingTitle) findingTitle.textContent = topPred.finding || 'Normal / No Finding';

  const icdTag = document.getElementById('viewerICDTag');
  if (icdTag) icdTag.textContent = topPred.icd10 ? `ICD-10: ${topPred.icd10}` : 'ICD-10: R91.8';

  const confTag = document.getElementById('viewerCalibratedConf');
  if (confTag) confTag.textContent = `${((topPred.probability || 0.85) * 100).toFixed(1)}%`;

  // Risk Gauge
  const riskVal = triage.risk_score || 0.2;
  const riskScoreEl = document.getElementById('viewerRiskScore');
  if (riskScoreEl) riskScoreEl.textContent = riskVal.toFixed(2);

  const riskBar = document.getElementById('viewerRiskBar');
  if (riskBar) {
    riskBar.style.width = `${Math.min(100, riskVal * 100)}%`;
    riskBar.className = `h-full ${riskVal >= 0.75 ? 'bg-red-500' : (riskVal >= 0.45 ? 'bg-amber-500' : 'bg-emerald-500')}`;
  }

  // Uncertainty Metrics
  const epistemicEl = document.getElementById('viewerEpistemic');
  if (epistemicEl) epistemicEl.textContent = triage.epistemic_uncertainty || '0.12';
  const aleatoricEl = document.getElementById('viewerAleatoric');
  if (aleatoricEl) aleatoricEl.textContent = triage.aleatoric_uncertainty || '0.06';

  // Segmentation Metrics
  const seg = ai.segmentation || {};
  const segAreaEl = document.getElementById('viewerSegArea');
  if (segAreaEl) segAreaEl.textContent = seg.has_segmentation ? `${seg.area_cm2} cm² (Dice: ${seg.dice_score})` : 'N/A';

  // Comparative Matrix
  renderComparativeMatrix(ai.baseline_comparison || []);

  // Pre-fill review form with AI suggestions
  const diagInput = document.getElementById('reviewFinalDiagnosis');
  if (diagInput) diagInput.value = topPred.finding || '';
  const indText = document.getElementById('reviewClinicalIndication');
  if (indText) indText.textContent = study.clinical_indication || 'Standard diagnostic radiograph.';
}

function renderComparativeMatrix(baselines) {
  const container = document.getElementById('baselineModelsMatrix');
  if (!container) return;

  container.innerHTML = baselines.map(m => {
    const isPrimary = m.model_id === state.activeModelId;
    return `
      <div class="p-2.5 rounded border ${isPrimary ? 'border-cyan-500/60 bg-cyan-950/20' : 'border-slate-700 bg-slate-800/40'} flex items-center justify-between text-xs">
        <div>
          <div class="font-bold text-slate-200 flex items-center gap-1.5">
            ${m.name}
            ${isPrimary ? '<span class="text-[10px] bg-cyan-500/20 text-cyan-300 px-1.5 py-0.5 rounded">Active</span>' : ''}
          </div>
          <div class="text-slate-400 text-[11px]">${m.architecture_family} • Latency: ${m.latency_ms}ms</div>
        </div>
        <div class="text-right">
          <div class="font-mono font-bold text-cyan-400">${(m.confidence * 100).toFixed(1)}%</div>
          <div class="text-[10px] text-slate-400 font-mono">AUC ${m.auc_roc}</div>
        </div>
      </div>
    `;
  }).join('');
}

// ==========================================
// 4. Viewer Interaction Controls (Window, Zoom, Pan)
// ==========================================
function initViewerControls() {
  // Presets
  const presetButtons = document.querySelectorAll('.window-preset-btn');
  presetButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      const preset = btn.dataset.preset;
      presetButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      await applyWindowPreset(preset);
    });
  });

  // Zoom buttons
  document.getElementById('zoomInBtn')?.addEventListener('click', () => {
    state.zoom = Math.min(state.zoom * 1.25, 5.0);
    renderCanvas();
    updateHUDZoom();
  });
  document.getElementById('zoomOutBtn')?.addEventListener('click', () => {
    state.zoom = Math.max(state.zoom / 1.25, 0.4);
    renderCanvas();
    updateHUDZoom();
  });
  document.getElementById('resetViewBtn')?.addEventListener('click', () => {
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
    state.isInverted = false;
    document.querySelectorAll('.window-preset-btn').forEach(button => {
      button.classList.toggle('active', button.dataset.preset === 'default');
    });
    applyWindowPreset('default');
    renderCanvas();
    updateHUDZoom();
  });
  document.getElementById('invertBtn')?.addEventListener('click', () => {
    state.isInverted = !state.isInverted;
    renderCanvas();
  });

  // Layer switches
  document.getElementById('toggleGradcam')?.addEventListener('change', (e) => {
    state.layers.gradcam = e.target.checked;
    renderCanvas();
  });
  document.getElementById('toggleBbox')?.addEventListener('change', (e) => {
    state.layers.bbox = e.target.checked;
    renderCanvas();
  });
  document.getElementById('toggleSegmentation')?.addEventListener('change', (e) => {
    state.layers.segmentation = e.target.checked;
    renderCanvas();
  });
  document.getElementById('gradcamAlphaSlider')?.addEventListener('input', (e) => {
    state.gradcamAlpha = parseFloat(e.target.value);
    renderCanvas();
  });
}

function updateHUDZoom() {
  const el = document.getElementById('hudBottomRight');
  if (el) {
    el.textContent = `Zoom: ${(state.zoom * 100).toFixed(0)}%`;
  }
}

async function applyWindowPreset(preset) {
  if (!state.selectedStudyId) return;
  try {
    const res = await fetch(`/studies/${state.selectedStudyId}/window`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset: preset, invert: state.isInverted })
    });
    if (!res.ok) throw new Error(`Window update failed (${res.status}).`);
    const data = await res.json();
    if (data.image_base64) {
      const img = new Image();
      img.onload = () => {
        state.cachedImages.base = img;
        renderCanvas();
      };
      img.src = data.image_base64;

      const hudBL = document.getElementById('hudBottomLeft');
      if (hudBL) hudBL.textContent = `W: ${data.window_width.toFixed(0)} L: ${data.window_center.toFixed(0)}`;
    }
  } catch (err) {
    console.error('Failed to apply window preset:', err);
    alert(`Could not change image contrast: ${err.message}`);
  }
}

function initCanvasInteraction() {
  const container = document.querySelector('.viewport-container');
  if (!container) return;

  container.addEventListener('mousedown', (e) => {
    state.isPanning = true;
    state.startMouseX = e.clientX - state.panX;
    state.startMouseY = e.clientY - state.panY;
  });

  window.addEventListener('mousemove', (e) => {
    if (!state.isPanning) return;
    state.panX = e.clientX - state.startMouseX;
    state.panY = e.clientY - state.startMouseY;
    renderCanvas();
  });

  window.addEventListener('mouseup', () => {
    state.isPanning = false;
  });

  container.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      state.zoom = Math.min(state.zoom * 1.1, 5.0);
    } else {
      state.zoom = Math.max(state.zoom / 1.1, 0.4);
    }
    renderCanvas();
    updateHUDZoom();
  });
}

// ==========================================
// 5. Radiologist Review & Feedback Actions
// ==========================================
function initReviewForm() {
  const form = document.getElementById('radiologistReviewForm');
  if (!form) return;

  const handleAction = async (actionType) => {
    if (!state.selectedStudyId) {
      alert('Please select a study from the worklist first.');
      return;
    }

    const finalDiag = document.getElementById('reviewFinalDiagnosis').value;
    const impressions = document.getElementById('reviewImpressions').value;
    const radName = document.getElementById('reviewDoctorName').value;
    if (!finalDiag.trim()) {
      alert('Enter a final diagnostic impression before signing off.');
      return;
    }

    const payload = {
      study_id: state.selectedStudyId,
      action: actionType,
      final_diagnosis: finalDiag,
      clinical_impressions: impressions || `Verified by ${radName} with action: ${actionType}`,
      radiologist_id: radName,
      override_rationale: actionType === 'ACCEPT' ? 'AI finding confirmed upon clinical review' : 'Clinician modified diagnosis'
    };

    const actionButtons = form.querySelectorAll('button[type="button"]');
    actionButtons.forEach(button => { button.disabled = true; });
    try {
      const res = await fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Review submission failed (${res.status}).`);
      if (data.status === 'SUCCESS') {
        showStructuredReportModal(data.feedback_record);
        await loadWorklist();
      }
    } catch (err) {
      console.error('Failed to submit review feedback:', err);
      alert(`Could not submit the review: ${err.message}`);
    } finally {
      actionButtons.forEach(button => { button.disabled = false; });
    }
  };

  document.getElementById('actionAcceptBtn')?.addEventListener('click', () => handleAction('ACCEPT'));
  document.getElementById('actionModifyBtn')?.addEventListener('click', () => handleAction('MODIFY'));
  document.getElementById('actionRejectBtn')?.addEventListener('click', () => handleAction('REJECT'));
}

function showStructuredReportModal(record) {
  const modal = document.getElementById('reportModal');
  const content = document.getElementById('reportModalContent');
  if (!modal || !content) return;

  content.innerHTML = `
    <div class="space-y-4 font-sans text-sm">
      <div class="p-3 bg-cyan-950/40 border border-cyan-800/60 rounded">
        <h4 class="font-bold text-cyan-300 text-base">DICOM Structured Report (TID 1500)</h4>
        <div class="text-xs text-slate-400">Study: ${escapeHtml(record.study_id)} • Signed: ${escapeHtml(record.timestamp)}</div>
      </div>
      <div class="grid grid-cols-2 gap-3 text-xs">
        <div><strong class="text-slate-400">Reviewing Physician:</strong> <div class="text-slate-200 font-semibold">${escapeHtml(record.radiologist_id)}</div></div>
        <div><strong class="text-slate-400">Workflow Action:</strong> <div class="text-emerald-400 font-bold font-mono">${escapeHtml(record.action)}</div></div>
        <div><strong class="text-slate-400">AI Preliminary Finding:</strong> <div class="text-slate-300">${escapeHtml(record.ai_original_finding)} (${(record.ai_original_confidence * 100).toFixed(0)}%)</div></div>
        <div><strong class="text-slate-400">Physician Final Diagnosis:</strong> <div class="text-cyan-300 font-bold">${escapeHtml(record.radiologist_final_diagnosis)}</div></div>
      </div>
      <div>
        <strong class="text-slate-400 text-xs">Clinical Impression:</strong>
        <p class="mt-1 p-2.5 bg-slate-800/80 rounded text-slate-200 text-xs leading-relaxed">${escapeHtml(record.clinical_impressions)}</p>
      </div>
      <div class="p-2.5 bg-slate-800/40 border border-slate-700 rounded text-[11px] text-slate-400 italic">
        Legal Attestation: ${escapeHtml(record.structured_report?.legal_attestation)}
      </div>
    </div>
  `;
  modal.classList.remove('hidden');
}

// ==========================================
// 6. DICOM & Medical Scan Upload
// ==========================================
function initUploadForm() {
  const uploadInput = document.getElementById('dicomFileInput');
  const uploadForm = document.getElementById('uploadScanForm');
  if (!uploadForm) return;

  uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = uploadInput?.files?.[0];
    if (!file) {
      alert('Please select a DICOM (.dcm) or standard medical image file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_id', state.activeModelId);
    formData.append('clinical_indication', document.getElementById('uploadIndication')?.value || 'Triage radiograph');

    const submitBtn = document.getElementById('uploadSubmitBtn');
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = 'Ingesting & Running AI...';
    }

    try {
      const res = await fetch('/studies', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        alert(`Study '${data.study_id}' ingested successfully!`);
        document.getElementById('uploadModal')?.classList.add('hidden');
        await loadWorklist();
        openStudyInViewer(data.study_id);
      } else {
        alert(`Upload error: ${data.detail || 'Failed to process file'}`);
      }
    } catch (err) {
      alert('Error uploading study: ' + err.message);
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Ingest & Run AI Model';
      }
    }
  });
}

// ==========================================
// 7. Models, MLOps Hub & Telemetry
// ==========================================
async function loadModels() {
  try {
    const res = await fetch('/models');
    const models = await res.json();
    const select = document.getElementById('headerModelSelect');
    if (select) {
      select.innerHTML = models.map(m => `
        <option value="${escapeHtml(m.id)}" ${m.is_active ? 'selected' : ''}>${escapeHtml(m.name)} (${escapeHtml(m.status)})</option>
      `).join('');
      state.activeModelId = select.value || state.activeModelId;

      select.addEventListener('change', (e) => {
        state.activeModelId = e.target.value;
      });
    }
  } catch (err) {
    console.error('Failed to load models:', err);
  }
}

async function initMLOps() {
  document.getElementById('triggerRetrainBtn')?.addEventListener('click', async () => {
    const btn = document.getElementById('triggerRetrainBtn');
    btn.disabled = true;
    btn.textContent = 'Starting Pipeline...';
    try {
      const res = await fetch('/mlops/retrain', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Request failed (${res.status}).`);
      alert('Demo retraining state updated. No model training was performed.');
      refreshMLOps();
    } catch (err) {
      alert('Failed: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Trigger Active-Learning Retraining';
    }
  });
}

async function refreshMLOps() {
  try {
    const res = await fetch('/mlops/metrics');
    if (!res.ok) throw new Error(`Monitoring request failed (${res.status}).`);
    const data = await res.json();

    const t = data.telemetry || {};
    const d = data.drift_analytics || {};
    const f = data.feedback_statistics || {};

    // Telemetry cards
    if (document.getElementById('mlopsP95Latency')) document.getElementById('mlopsP95Latency').textContent = `${t.p95_latency_ms} ms`;
    if (document.getElementById('mlopsThroughput')) document.getElementById('mlopsThroughput').textContent = `${t.requests_processed} reqs`;
    if (document.getElementById('mlopsGPU')) document.getElementById('mlopsGPU').textContent = `${t.gpu_utilization_percent}% (${t.gpu_memory_used_gb} / ${t.gpu_memory_total_gb} GB)`;
    if (document.getElementById('mlopsAgreement')) document.getElementById('mlopsAgreement').textContent = `${f.concordance_rate}%`;

    // Drift Indicator
    const driftBadge = document.getElementById('driftStatusBadge');
    if (driftBadge) {
      driftBadge.textContent = d.drift_status;
      driftBadge.className = `clinical-badge ${d.drift_status === 'NORMAL' ? 'badge-routine' : 'badge-urgent'}`;
    }
    if (document.getElementById('psiScoreVal')) document.getElementById('psiScoreVal').textContent = d.population_stability_index;
    if (document.getElementById('ksScoreVal')) document.getElementById('ksScoreVal').textContent = d.ks_statistic_p_value;

    // Registry Table
    const registryTable = document.getElementById('modelRegistryBody');
    if (registryTable && data.models) {
      registryTable.innerHTML = data.models.map(m => `
        <tr>
          <td class="font-bold text-slate-200">${m.name}</td>
          <td><span class="px-2 py-0.5 rounded text-xs ${m.status === 'Production' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50' : 'bg-slate-700 text-slate-300'} font-semibold">${m.status}</span></td>
          <td class="font-mono text-cyan-400 font-bold">${m.auc_roc || m.dice_score || '--'}</td>
          <td class="font-mono text-slate-300">${m.latency_ms} ms</td>
          <td class="text-xs text-slate-400">${m.framework}</td>
          <td>
            ${m.status !== 'Production' ? `
              <button class="tool-btn text-xs py-1 px-2.5 text-cyan-300 border-cyan-800" onclick="promoteModel('${m.id}')">Promote to Prod</button>
            ` : '<span class="text-xs text-emerald-400 font-semibold">Active Engine</span>'}
          </td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.error('Failed to refresh MLOps:', err);
    alert(`Could not refresh demo monitoring: ${err.message}`);
  }
}

async function promoteModel(modelId) {
  const fd = new FormData();
  fd.append('model_id', modelId);
  try {
    const response = await fetch('/mlops/promote', { method: 'POST', body: fd });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || `Request failed (${response.status}).`);
    alert(`Demo catalog updated for '${modelId}'. No production model was deployed.`);
    await loadModels();
    refreshMLOps();
  } catch (err) {
    alert('Failed to promote: ' + err.message);
  }
}

// ==========================================
// 8. Data Quality & Patient Leakage Auditor
// ==========================================
function initDataQuality() {
  document.getElementById('runLeakageCheckBtn')?.addEventListener('click', async () => {
    // Demonstration patient split cohorts
    const payload = {
      train_ids: ["PT-94021", "PT-31950", "PT-77142", "PT-11204", "PT-55931", "PT-88320"],
      val_ids: ["PT-60293", "PT-44109"],
      test_ids: ["PT-19482", "PT-66381"]
    };

    try {
      const res = await fetch('/quality/leakage-check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Audit failed (${res.status}).`);
      const output = document.getElementById('leakageAuditOutput');
      if (output) {
        output.innerHTML = `
          <div class="p-4 rounded border ${data.leakage_free ? 'border-emerald-500/50 bg-emerald-950/20 text-emerald-300' : 'border-red-500/50 bg-red-950/20 text-red-300'} font-sans text-xs space-y-2">
            <div class="font-bold text-sm">${data.verdict}</div>
            <div>Unique Patients: <strong>${data.total_patients}</strong> (Train: ${data.train_unique_patients}, Val: ${data.val_unique_patients}, Test: ${data.test_unique_patients})</div>
            <div>Train ∩ Val Overlap: <strong>${data.leakage_train_val.length} patients</strong></div>
            <div>Train ∩ Test Overlap: <strong>${data.leakage_train_test.length} patients</strong></div>
          </div>
        `;
      }
    } catch (err) {
      alert('Error: ' + err.message);
    }
  });
}
