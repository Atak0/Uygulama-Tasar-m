/**
 * SpineAI - Omurga X-Ray Analiz Sistemi
 * Frontend Application Logic
 */

const API = {
    analyze: '/api/analyze',
    history: '/api/history',
    stats: '/api/stats',
    analysis: (id) => `/api/analysis/${encodeURIComponent(id)}`,
    report: (id) => `/api/analysis/${encodeURIComponent(id)}/report.pdf`,
    updateProfile: '/api/auth/update',
};

const DISPLAY = {
    normal: 'Normal',
    kayma: 'Bel Kayması',
    skolyoz: 'Skolyoz',
};

// -------------------------------------------
// STATE
// -------------------------------------------
let selectedFile = null;
let currentPage = 'dashboard';
let lightboxImages = [];
let lightboxIndex = 0;

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[ch]));
}

function escapeAttr(value) {
    return escapeHtml(value).replace(/`/g, '&#96;');
}

function safeClass(value, fallback = 'normal') {
    const normalized = String(value || '').toLowerCase();
    return /^[a-z0-9_-]+$/.test(normalized) ? normalized : fallback;
}

// -------------------------------------------
// INITIALIZATION
// -------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initUpload();
    initImageTabs();
    loadDashboard();

    document.getElementById('btnNewAnalysis')?.addEventListener('click', () => {
        clearFormAndResults();
        navigateTo('analysis');
    });
    document.getElementById('btnGoToAnalysis')?.addEventListener('click', () => {
        clearFormAndResults();
        navigateTo('analysis');
    });
    document.getElementById('menuToggle')?.addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
    });
    document.getElementById('logoHome')?.addEventListener('click', () => navigateTo('dashboard'));
    document.getElementById('logoHome')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            navigateTo('dashboard');
        }
    });
    document.getElementById('btnFinishAnalysis')?.addEventListener('click', finishAnalysis);
    document.getElementById('doctorProfile')?.addEventListener('click', openProfileModal);
    document.getElementById('doctorProfile')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            openProfileModal();
        }
    });
    document.getElementById('profileForm')?.addEventListener('submit', saveProfileSettings);
    document.querySelectorAll('.password-toggle').forEach(btn => btn.addEventListener('click', togglePasswordVisibility));

    // Logout
    document.getElementById('btnLogout')?.addEventListener('click', async () => {
        try {
            await fetch('/api/auth/logout', { method: 'POST' });
        } catch (e) { /* ignore */ }
        window.location.href = 'login.html';
    });
});

// -------------------------------------------
// NAVIGATION
// -------------------------------------------
function initNavigation() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            navigateTo(page);
        });
    });
}

function navigateTo(page) {
    if (page === 'analysis') {
        clearFormAndResults();
    }
    currentPage = page;
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelector(`[data-page="${page}"]`)?.classList.add('active');
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${page}`)?.classList.add('active');
    document.getElementById('sidebar')?.classList.remove('open');

    if (page === 'dashboard') loadDashboard();
    if (page === 'history') loadHistory();
}

// -------------------------------------------
// DASHBOARD
// -------------------------------------------
async function loadDashboard() {
    try {
        const [statsRes, histRes] = await Promise.all([
            fetch(API.stats).then(r => r.json()),
            fetch(API.history).then(r => r.json()),
        ]);

        if (statsRes.success) {
            const s = statsRes.data;
            document.getElementById('statTotal').textContent = s.total_analyses;
            document.getElementById('statNormal').textContent = s.class_distribution.normal || 0;
            document.getElementById('statKayma').textContent = s.class_distribution.kayma || 0;
            document.getElementById('statSkolyoz').textContent = s.class_distribution.skolyoz || 0;
            renderDistribution(s);
        }

        if (histRes.success) {
            renderRecentList(histRes.data.slice(0, 5));
        }
    } catch (e) {
        console.error('Dashboard load error:', e);
    }
}

function renderDistribution(stats) {
    const el = document.getElementById('distributionChart');
    const total = stats.total_analyses || 1;
    if (total === 0 || (!stats.class_distribution.normal && !stats.class_distribution.kayma && !stats.class_distribution.skolyoz)) {
        el.innerHTML = '<div class="empty-state-mini">Henüz analiz yapılmadı</div>';
        return;
    }
    const classes = [
        { key: 'normal', label: 'Normal', count: stats.class_distribution.normal || 0 },
        { key: 'kayma', label: 'Bel Kayması', count: stats.class_distribution.kayma || 0 },
        { key: 'skolyoz', label: 'Skolyoz', count: stats.class_distribution.skolyoz || 0 },
    ];
    el.innerHTML = classes.map(c => {
        const pct = ((c.count / total) * 100).toFixed(0);
        return `<div class="dist-row">
            <div class="dist-label">${c.label}</div>
            <div class="dist-bar-wrapper">
                <div class="dist-bar ${c.key}" style="width:${Math.max(pct, 3)}%">${c.count}</div>
            </div>
        </div>`;
    }).join('');
}

function renderRecentList(analyses) {
    const el = document.getElementById('recentList');
    if (!analyses.length) {
        el.innerHTML = '<div class="empty-state-mini">Henüz analiz yapılmadı</div>';
        return;
    }
    el.innerHTML = analyses.map(a => {
        const cls = safeClass(a.prediction?.class);
        const name = escapeHtml(a.patient_name || a.patient_id || 'Anonim');
        const date = new Date(a.timestamp).toLocaleDateString('tr-TR', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
        const conf = ((a.prediction?.confidence || 0) * 100).toFixed(1);
        return `<div class="recent-item">
            <div class="recent-dot ${cls}"></div>
            <div class="recent-info">
                <div class="recent-name">${name}</div>
                <div class="recent-date">${date}</div>
            </div>
            <div class="recent-conf">%${conf}</div>
        </div>`;
    }).join('');
}

// -------------------------------------------
// FILE UPLOAD
// -------------------------------------------
function initUpload() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const previewRemove = document.getElementById('previewRemove');

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });
    previewRemove.addEventListener('click', clearFile);
    btnAnalyze.addEventListener('click', runAnalysis);
}

function handleFile(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/bmp', 'image/tiff', 'image/webp'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(png|jpe?g|bmp|tiff|webp)$/i)) {
        showToast('Desteklenmeyen dosya formatı', 'error');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
        document.getElementById('previewFilename').textContent = file.name;
        document.getElementById('previewSize').textContent = formatBytes(file.size);
        document.getElementById('dropZone').style.display = 'none';
        document.getElementById('previewArea').style.display = 'block';
        document.getElementById('btnAnalyze').disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearFile() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('dropZone').style.display = 'block';
    document.getElementById('previewArea').style.display = 'none';
    document.getElementById('btnAnalyze').disabled = true;
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function clearFormAndResults() {
    clearFile();
    if(document.getElementById('patientId')) document.getElementById('patientId').value = '';
    if(document.getElementById('patientName')) document.getElementById('patientName').value = '';
    if(document.getElementById('clinicalNotes')) document.getElementById('clinicalNotes').value = '';
    if(document.getElementById('modelType')) document.getElementById('modelType').value = 'multiclass';
    if(document.getElementById('resultsPanel')) document.getElementById('resultsPanel').style.display = 'none';
    if(document.getElementById('loadingCard')) document.getElementById('loadingCard').style.display = 'none';
    if(document.getElementById('resultContent')) document.getElementById('resultContent').style.display = 'none';
    if(document.getElementById('confidenceBar')) document.getElementById('confidenceBar').style.width = '0';
    document.querySelectorAll('.image-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.image-view').forEach(v => v.classList.remove('active'));
    document.querySelector('.image-tab')?.classList.add('active');
    document.getElementById('viewOriginal')?.classList.add('active');
}

function finishAnalysis() {
    clearFormAndResults();
    document.getElementById('uploadCard')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    showToast('Analiz ekranı sıfırlandı', 'info');
}


// -------------------------------------------
// ANALYSIS
// -------------------------------------------
async function runAnalysis() {
    if (!selectedFile) return;

    const btn = document.getElementById('btnAnalyze');
    btn.disabled = true;
    btn.innerHTML = '<span>Analiz ediliyor...</span>';

    const resultsPanel = document.getElementById('resultsPanel');
    const loadingCard = document.getElementById('loadingCard');
    const resultContent = document.getElementById('resultContent');

    resultsPanel.style.display = 'block';
    loadingCard.style.display = 'block';
    resultContent.style.display = 'none';

    // Scroll to results
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('patient_id', document.getElementById('patientId').value);
    formData.append('patient_name', document.getElementById('patientName').value);
    formData.append('notes', document.getElementById('clinicalNotes').value);
    formData.append('model_type', document.getElementById('modelType').value);

    try {
        const res = await fetch(API.analyze, { method: 'POST', body: formData });
        const data = await res.json();

        if (!res.ok || data.error) {
            throw new Error(data.error || 'Analiz sırasında hata oluştu');
        }

        renderResults(data.data);
        showToast('Analiz tamamlandı', 'success');
    } catch (e) {
        showToast(e.message, 'error');
        resultsPanel.style.display = 'none';
    } finally {
        loadingCard.style.display = 'none';
        btn.disabled = false;
        btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg> Analizi Başlat`;
    }
}

function renderResults(result) {
    const pred = result.prediction;
    const severity = pred.severity;
    const conf = (pred.confidence * 100).toFixed(1);

    // Diagnosis Card
    const diagCard = document.getElementById('diagnosisCard');
    diagCard.className = `card diagnosis-card severity-${severity}`;

    const icon = document.getElementById('diagnosisIcon');
    icon.className = `diagnosis-icon severity-${severity}`;
    icon.innerHTML = pred.class === 'normal' ? '&#10003;' : '&#9888;';

    document.getElementById('diagnosisName').textContent = pred.display_name;
    document.getElementById('diagnosisDesc').textContent = pred.description;
    document.getElementById('confidenceText').textContent = `%${conf}`;

    const confBar = document.getElementById('confidenceBar');
    const confLevel = pred.confidence >= 0.85 ? 'high' : pred.confidence >= 0.6 ? 'medium' : 'low';
    confBar.className = `confidence-bar ${confLevel}`;
    setTimeout(() => { confBar.style.width = `${conf}%`; }, 100);

    const warning = document.getElementById('diagnosisWarning');
    warning.style.display = pred.is_uncertain ? 'flex' : 'none';

    // Probability Bars
    const probBars = document.getElementById('probBars');
    probBars.innerHTML = Object.entries(pred.probabilities)
        .sort((a, b) => b[1] - a[1])
        .map(([cls, prob]) => {
            const pct = (prob * 100).toFixed(1);
            const clsName = safeClass(cls);
            return `<div class="prob-row">
                <div class="prob-label">${escapeHtml(DISPLAY[cls] || cls)}</div>
                <div class="prob-bar-wrapper">
                    <div class="prob-bar ${clsName}" style="width:${Math.max(pct, 2)}%">${pct > 10 ? `%${pct}` : ''}</div>
                </div>
                <div class="prob-value">%${pct}</div>
            </div>`;
        }).join('');

    // Images
    if (result.images) {
        document.getElementById('imgOriginal').src = `data:image/png;base64,${result.images.original}`;
        document.getElementById('imgProcessed').src = `data:image/png;base64,${result.images.processed}`;
        document.getElementById('imgHeatmap').src = `data:image/png;base64,${result.images.heatmap}`;
        document.getElementById('imgOverlay').src = `data:image/png;base64,${result.images.overlay}`;
    }

    // Reset to first tab
    document.querySelectorAll('.image-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.image-view').forEach(v => v.classList.remove('active'));
    document.querySelector('.image-tab')?.classList.add('active');
    document.getElementById('viewOriginal')?.classList.add('active');

    // Meta Grid
    const meta = document.getElementById('metaGrid');
    meta.innerHTML = `
        <div class="meta-item"><div class="meta-label">Analiz ID</div><div class="meta-value">${result.id}</div></div>
        <div class="meta-item"><div class="meta-label">Tarih / Saat</div><div class="meta-value">${new Date(result.timestamp).toLocaleString('tr-TR')}</div></div>
        <div class="meta-item"><div class="meta-label">Hasta ID</div><div class="meta-value">${escapeHtml(result.patient_id || '-')}</div></div>
        <div class="meta-item"><div class="meta-label">Hasta Adı</div><div class="meta-value">${escapeHtml(result.patient_name || '-')}</div></div>
        <div class="meta-item"><div class="meta-label">Dosya</div><div class="meta-value">${escapeHtml(result.image_filename)}</div></div>
        <div class="meta-item"><div class="meta-label">Conv Katmanı</div><div class="meta-value" style="font-family:var(--font-mono);font-size:.78rem">${escapeHtml(result.gradcam?.conv_layer || '-')}</div></div>
        <div class="meta-item"><div class="meta-label">Güven Marjı</div><div class="meta-value">${(pred.margin * 100).toFixed(1)}%</div></div>
        <div class="meta-item"><div class="meta-label">Model</div><div class="meta-value">EfficientNetB0</div></div>
    `;

    document.getElementById('resultContent').style.display = 'block';
    document.getElementById('resultContent').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// -------------------------------------------
// IMAGE TABS
// -------------------------------------------
function initImageTabs() {
    document.querySelectorAll('.image-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.image-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.image-view').forEach(v => v.classList.remove('active'));
            tab.classList.add('active');
            const targetId = tab.dataset.target;
            document.getElementById(targetId)?.classList.add('active');
            
            const legend = document.getElementById('gradcamLegend');
            if(legend) {
                if(targetId === 'viewHeatmap' || targetId === 'viewOverlay') {
                    legend.style.display = 'block';
                } else {
                    legend.style.display = 'none';
                }
            }
        });
    });
}

// -------------------------------------------
// HISTORY
// -------------------------------------------
async function loadHistory() {
    try {
        const res = await fetch(API.history);
        const data = await res.json();
        if (data.success) renderHistory(data.data);
    } catch (e) {
        console.error('History load error:', e);
    }
}

function renderHistory(analyses) {
    const body = document.getElementById('historyBody');
    const empty = document.getElementById('historyEmpty');
    const table = document.getElementById('historyTable');

    if (!analyses.length) {
        table.style.display = 'none';
        empty.style.display = 'block';
        return;
    }
    table.style.display = 'table';
    empty.style.display = 'none';

    body.innerHTML = analyses.map(a => {
        const pred = a.prediction || {};
        const cls = safeClass(pred.class);
        const date = new Date(a.timestamp).toLocaleString('tr-TR', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
        const conf = ((pred.confidence || 0) * 100).toFixed(1);
        const statusClass = pred.is_uncertain ? 'uncertain' : 'confident';
        const statusText = pred.is_uncertain ? 'Belirsiz' : 'Güvenilir';
        return `<tr class="history-row" data-analysis-id="${escapeAttr(a.id)}" style="cursor:pointer;">
            <td>${date}</td>
            <td>${escapeHtml(a.patient_id || '-')}</td>
            <td>${escapeHtml(a.patient_name || '-')}</td>
            <td><span class="badge badge-${cls}">${escapeHtml(DISPLAY[cls] || cls)}</span></td>
            <td style="font-family:var(--font-mono);font-weight:600">%${conf}</td>
            <td><span class="badge badge-${statusClass}">${statusText}</span></td>
            <td><button class="btn btn-ghost btn-sm history-delete" data-analysis-id="${escapeAttr(a.id)}">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:16px;height:16px"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button></td>
        </tr>`;
    }).join('');

    body.querySelectorAll('.history-row').forEach(row => {
        row.addEventListener('click', () => showAnalysisDetail(row.dataset.analysisId));
    });
    body.querySelectorAll('.history-delete').forEach(btn => {
        btn.addEventListener('click', (event) => {
            event.stopPropagation();
            deleteAnalysis(btn.dataset.analysisId);
        });
    });

    // Search filter
    const searchInput = document.getElementById('historySearch');
    searchInput.oninput = () => {
        const q = searchInput.value.toLowerCase();
        body.querySelectorAll('tr').forEach(row => {
            row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
        });
    };
}

async function deleteAnalysis(id) {
    if (!confirm('Bu analizi silmek istediğinize emin misiniz?')) return;
    try {
        await fetch(API.analysis(id), { method: 'DELETE' });
        showToast('Analiz silindi', 'info');
        loadHistory();
        loadDashboard();
    } catch (e) {
        showToast('Silme hatası', 'error');
    }
}

// -------------------------------------------
// DETAILS MODAL
// -------------------------------------------
async function showAnalysisDetail(id) {
    try {
        const res = await fetch(API.analysis(id));
        const data = await res.json();
        if(!data.success) {
            showToast('Detaylar yüklenemedi', 'error');
            return;
        }
        
        const a = data.data;
        const modal = document.getElementById('detailModal');
        const modalBody = document.getElementById('modalBody');
        
        const pred = a.prediction || {};
        const cls = safeClass(pred.class);
        const date = new Date(a.timestamp).toLocaleString('tr-TR');
        
        const imageTiles = [];
        if (a.images) {
            imageTiles.push(
                { label: 'Orijinal', src: `data:image/png;base64,${a.images.original}` },
                { label: 'Ön İşlenmiş', src: `data:image/png;base64,${a.images.processed}` },
                { label: 'Isı Haritası', src: `data:image/png;base64,${a.images.heatmap}` },
                { label: 'Bindirme', src: `data:image/png;base64,${a.images.overlay}` },
            );
        } else if (a.image_urls) {
            if (a.image_urls.original) imageTiles.push({ label: 'Orijinal', src: a.image_urls.original });
            if (a.image_urls.processed) imageTiles.push({ label: 'Ön İşlenmiş', src: a.image_urls.processed });
            if (a.image_urls.heatmap) imageTiles.push({ label: 'Isı Haritası', src: a.image_urls.heatmap });
            if (a.image_urls.overlay) imageTiles.push({ label: 'Bindirme', src: a.image_urls.overlay });
        }

        let imagesHtml = '';
        if(imageTiles.length) {
            lightboxImages = imageTiles;
            imagesHtml = `
            <div style="margin-top:20px; margin-bottom:10px;">
                <h4 style="margin-bottom:10px;">Görseller</h4>
                <div class="detail-images">
                    ${imageTiles.map((img, index) => `
                        <div class="detail-image-item">
                            <button type="button" class="detail-image-button" data-image-index="${index}">
                                <img src="${escapeAttr(img.src)}" alt="${escapeAttr(img.label)}">
                            </button>
                            <div>${escapeHtml(img.label)}</div>
                        </div>
                    `).join('')}
                </div>
            </div>`;
        } else {
            lightboxImages = [];
        }
        
        modalBody.innerHTML = `
            <div style="display:flex; flex-direction:column; gap:16px;">
                <div class="detail-actions">
                    <a class="btn btn-outline btn-sm" href="${escapeAttr(API.report(a.id))}" target="_blank" rel="noopener">
                        PDF Rapor İndir
                    </a>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                    <div><span style="color:var(--text-muted); font-size:0.85rem;">Tarih</span><br><strong>${date}</strong></div>
                    <div><span style="color:var(--text-muted); font-size:0.85rem;">Hasta</span><br><strong>${escapeHtml(a.patient_name || 'Anonim')} (${escapeHtml(a.patient_id || '-')})</strong></div>
                    <div><span style="color:var(--text-muted); font-size:0.85rem;">Tanı</span><br><span class="badge badge-${cls}">${DISPLAY[cls] || cls}</span></div>
                    <div><span style="color:var(--text-muted); font-size:0.85rem;">Güven</span><br><strong>%${((pred.confidence||0)*100).toFixed(1)}</strong></div>
                </div>
                <div>
                    <span style="color:var(--text-muted); font-size:0.85rem;">Model</span><br><strong>${escapeHtml(a.model_type || 'multiclass')}</strong>
                </div>
                ${a.notes ? `<div><span style="color:var(--text-muted); font-size:0.85rem;">Klinik Notlar</span><br><div style="padding:10px; background:var(--bg-input); border-radius:8px; margin-top:4px; font-size:0.9rem;">${escapeHtml(a.notes)}</div></div>` : ''}
                ${imagesHtml}
            </div>
        `;
        modalBody.querySelectorAll('.detail-image-button').forEach(btn => {
            btn.addEventListener('click', () => openImageLightbox(Number(btn.dataset.imageIndex)));
        });
        
        modal.style.display = 'flex';
        
    } catch (e) {
        showToast('Detay hatası', 'error');
        console.error(e);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('closeModal')?.addEventListener('click', () => {
        document.getElementById('detailModal').style.display = 'none';
    });
    document.getElementById('detailModal')?.addEventListener('click', (e) => {
        if(e.target.id === 'detailModal') {
            e.target.style.display = 'none';
        }
    });
    document.getElementById('closeProfileModal')?.addEventListener('click', closeProfileModal);
    document.getElementById('profileModal')?.addEventListener('click', (e) => {
        if (e.target.id === 'profileModal') closeProfileModal();
    });
    document.getElementById('closeImageLightbox')?.addEventListener('click', closeImageLightbox);
    document.getElementById('imageLightbox')?.addEventListener('click', (e) => {
        if (e.target.id === 'imageLightbox') closeImageLightbox();
    });
    document.getElementById('lightboxPrev')?.addEventListener('click', (e) => {
        e.stopPropagation();
        showLightboxImage(lightboxIndex - 1);
    });
    document.getElementById('lightboxNext')?.addEventListener('click', (e) => {
        e.stopPropagation();
        showLightboxImage(lightboxIndex + 1);
    });
    document.addEventListener('keydown', (e) => {
        const lightboxOpen = document.getElementById('imageLightbox')?.style.display === 'flex';
        if (!lightboxOpen) return;
        if (e.key === 'Escape') closeImageLightbox();
        if (e.key === 'ArrowLeft') showLightboxImage(lightboxIndex - 1);
        if (e.key === 'ArrowRight') showLightboxImage(lightboxIndex + 1);
    });
});

// -------------------------------------------
// PROFILE SETTINGS
// -------------------------------------------
function populateProfileForm(user) {
    document.getElementById('profileFullName') && (document.getElementById('profileFullName').value = user.full_name || '');
    document.getElementById('profileTitle') && (document.getElementById('profileTitle').value = user.title || '');
    document.getElementById('profileEmail') && (document.getElementById('profileEmail').value = user.email || '');
    document.getElementById('profileUsername') && (document.getElementById('profileUsername').value = user.username || '');
}

function openProfileModal() {
    if (window.currentUser) populateProfileForm(window.currentUser);
    ['profileCurrentPassword', 'profileNewPassword', 'profileConfirmPassword'].forEach(id => {
        const input = document.getElementById(id);
        if (input) {
            input.value = '';
            input.type = 'password';
        }
    });
    document.querySelectorAll('.password-toggle').forEach(btn => btn.textContent = 'Göster');
    const error = document.getElementById('profileError');
    if (error) error.style.display = 'none';
    document.getElementById('profileModal').style.display = 'flex';
}

function closeProfileModal() {
    document.getElementById('profileModal').style.display = 'none';
}

function togglePasswordVisibility(e) {
    const btn = e.currentTarget;
    const input = document.getElementById(btn.dataset.target);
    if (!input) return;
    const isHidden = input.type === 'password';
    input.type = isHidden ? 'text' : 'password';
    btn.textContent = isHidden ? 'Gizle' : 'Göster';
}

async function saveProfileSettings(e) {
    e.preventDefault();
    const error = document.getElementById('profileError');
    const btn = document.getElementById('btnSaveProfile');
    error.style.display = 'none';

    const full_name = document.getElementById('profileFullName').value.trim();
    const title = document.getElementById('profileTitle').value.trim();
    const email = document.getElementById('profileEmail').value.trim();
    const current_password = document.getElementById('profileCurrentPassword').value;
    const new_password = document.getElementById('profileNewPassword').value;
    const confirm_password = document.getElementById('profileConfirmPassword').value;

    if (!full_name || !email) {
        error.textContent = 'Ad soyad ve e-posta alanları boş bırakılamaz.';
        error.style.display = 'block';
        return;
    }
    if (new_password || confirm_password || current_password) {
        if (!current_password) {
            error.textContent = 'Şifre değiştirmek için mevcut şifrenizi girin.';
            error.style.display = 'block';
            return;
        }
        if (new_password.length < 6) {
            error.textContent = 'Yeni şifre en az 6 karakter olmalıdır.';
            error.style.display = 'block';
            return;
        }
        if (new_password !== confirm_password) {
            error.textContent = 'Yeni şifreler eşleşmiyor.';
            error.style.display = 'block';
            return;
        }
    }

    btn.disabled = true;
    btn.textContent = 'Kaydediliyor...';
    try {
        const payload = { full_name, title, email };
        if (new_password) {
            payload.current_password = current_password;
            payload.new_password = new_password;
        }
        const res = await fetch(API.updateProfile, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok || !data.success) throw new Error(data.error || 'Profil güncellenemedi.');
        window.currentUser = data.user;
        document.getElementById('sidebarDoctorName').textContent = data.user.full_name || data.user.username;
        document.getElementById('sidebarDoctorTitle').textContent = data.user.title || 'Doktor';
        closeProfileModal();
        showToast('Hesap ayarları güncellendi', 'success');
    } catch (err) {
        error.textContent = err.message;
        error.style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Değişiklikleri Kaydet';
    }
}

window.populateProfileForm = populateProfileForm;

// -------------------------------------------
// IMAGE LIGHTBOX
// -------------------------------------------
function openImageLightbox(index = 0) {
    showLightboxImage(index);
    const modal = document.getElementById('imageLightbox');
    modal.style.display = 'flex';
}

function showLightboxImage(index) {
    if (!lightboxImages.length) return;
    lightboxIndex = (index + lightboxImages.length) % lightboxImages.length;
    const item = lightboxImages[lightboxIndex];
    document.getElementById('lightboxImage').src = item.src;
    document.getElementById('lightboxImage').alt = item.label;
    document.getElementById('lightboxLabel').textContent = `${item.label} (${lightboxIndex + 1}/${lightboxImages.length})`;
}

function closeImageLightbox() {
    const modal = document.getElementById('imageLightbox');
    modal.style.display = 'none';
    document.getElementById('lightboxImage').src = '';
}


// -------------------------------------------
// TOAST
// -------------------------------------------
function showToast(msg, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3500);
}
