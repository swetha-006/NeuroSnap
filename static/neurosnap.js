/* neurosnap.js */
(function () {
  'use strict';

  /* ── Drag-and-drop upload zone ── */
  const zone  = document.getElementById('uploadZone');
  const input = document.getElementById('imageInput');
  const wrap  = document.getElementById('previewWrap');
  const img   = document.getElementById('previewImg');
  const hint  = document.getElementById('uploadHint');

  function showPreview(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = function (e) {
      img.src = e.target.result;
      wrap.classList.add('visible');
      if (hint) hint.classList.add('hidden');
    };
    reader.readAsDataURL(file);
  }

  if (zone && input) {
    zone.addEventListener('click', function (e) {
      if (e.target !== input) input.click();
    });

    input.addEventListener('change', function () {
      if (this.files[0]) showPreview(this.files[0]);
    });

    ['dragenter', 'dragover'].forEach(function (ev) {
      zone.addEventListener(ev, function (e) {
        e.preventDefault();
        zone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(function (ev) {
      zone.addEventListener(ev, function (e) {
        e.preventDefault();
        zone.classList.remove('dragover');
      });
    });

    zone.addEventListener('drop', function (e) {
      const file = e.dataTransfer.files[0];
      if (file) {
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        showPreview(file);
      }
    });
  }

  /* ── Analyse button spinner ── */
  const form = document.getElementById('uploadForm');
  const btn  = document.getElementById('analyseBtn');

  if (form && btn) {
    form.addEventListener('submit', function () {
      btn.disabled = true;
      btn.innerHTML =
        '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analysing…';
    });
  }

  /* ── Confidence gauge animation ── */
  /* FIX: use requestAnimationFrame + double-rAF so the browser has painted
     the initial dashoffset=345 before we transition to the target value.
     The old setTimeout(120ms) sometimes fired before the CSS transition
     was registered, resulting in no animation or a jump to 100%. */
  const gaugeFill = document.getElementById('gaugeFill');
  if (gaugeFill) {
    const pct           = parseFloat(gaugeFill.dataset.pct) || 0;
    const circumference = 2 * Math.PI * 55;          // 2πr where r=55 → ≈345.58
    gaugeFill.setAttribute('stroke-dasharray',  circumference);
    gaugeFill.setAttribute('stroke-dashoffset', circumference); // start hidden
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        var offset = circumference - (pct / 100) * circumference;
        gaugeFill.setAttribute('stroke-dashoffset', offset);
      });
    });
  }

  /* ── Auto-dismiss flash alerts ── */
  document.querySelectorAll('.ns-alerts .alert').forEach(function (el) {
    setTimeout(function () {
      var bsAlert = bootstrap.Alert.getOrCreateInstance(el);
      bsAlert.close();
    }, 5000);
  });

})();
