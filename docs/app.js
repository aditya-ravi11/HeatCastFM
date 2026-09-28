// HeatCast-FM static viewer: reads the exported backtest forecasts from data/*.json.
(function () {
  "use strict";

  const LEVELS = [
    { name: "Green", text: "No warning" },
    { name: "Yellow", text: "Heat watch: be aware" },
    { name: "Orange", text: "Heatwave alert: be prepared" },
    { name: "Red", text: "Severe heatwave: take action" },
  ];
  const HISTORY = 20;
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

  const $ = (id) => document.getElementById(id);
  const els = {
    region: $("region"), model: $("model"), date: $("date"), note: $("note"), banner: $("banner"),
    chart: $("chart"), legend: $("legend"), tooltip: $("tooltip"), tbody: document.querySelector("#table tbody"),
    level: $("s-level"), hw: $("s-hw"), hot: $("s-hot"), rule: $("rule"),
  };

  let meta = null;
  const cache = new Map();

  const t10 = (v) => (v === null || v === undefined ? null : v / 10);
  const fmt1 = (v) => (v === null || v === undefined ? "" : v.toFixed(1));
  const parseDay = (s) => { const [y, m, d] = s.split("-").map(Number); return new Date(Date.UTC(y, m - 1, d)); };
  const shortDate = (s) => { const d = parseDay(s); return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`; };
  const longDate = (s) => { const d = parseDay(s); return `${WEEKDAYS[d.getUTCDay()]} ${String(d.getUTCDate()).padStart(2, "0")} ${MONTHS[d.getUTCMonth()]}`; };

  async function loadJSON(url) {
    if (cache.has(url)) return cache.get(url);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not load ${url}`);
    const data = await res.json();
    cache.set(url, data);
    return data;
  }

  function regionMeta(name) { return meta.regions.find((r) => r.name === name); }
  function modelMeta(name) { return meta.models.find((m) => m.name === name); }

  function level(pHot, pHw, pSev, th) {
    let l = 0;
    if (pHot >= th.t_watch * 100) l = 1;
    if (pHw >= th.t_alert * 100) l = 2;
    if (pSev >= th.t_severe * 100) l = 3;
    return l;
  }

  function nearestOrigin(origins, dayIdx) {
    let best = origins[0];
    for (const o of origins) if (Math.abs(o - dayIdx) < Math.abs(best - dayIdx)) best = o;
    return best;
  }

  function dayIndexOf(dateStr) {
    const i = meta.days.indexOf(dateStr);
    if (i >= 0) return i;
    // fall back to the closest stored day
    const target = parseDay(dateStr).getTime();
    let best = 0, bestGap = Infinity;
    meta.days.forEach((d, k) => { const g = Math.abs(parseDay(d).getTime() - target); if (g < bestGap) { bestGap = g; best = k; } });
    return best;
  }

  function syncUrl(region, model, date) {
    const p = new URLSearchParams({ region, model, date });
    history.replaceState(null, "", `?${p.toString()}`);
  }

  async function render() {
    const regionName = els.region.value;
    const modelName = els.model.value;
    const reg = regionMeta(regionName);
    const mod = modelMeta(modelName);
    let data;
    try {
      data = await loadJSON(`data/${reg.slug}__${mod.slug}.json`);
    } catch (e) {
      els.note.hidden = false;
      els.note.textContent = `${modelName} was not run for ${regionName}.`;
      return;
    }

    const wanted = dayIndexOf(els.date.value);
    const origin = data.origins.includes(wanted) ? wanted : nearestOrigin(data.origins, wanted);
    const issueDate = meta.days[origin];
    if (origin !== wanted) {
      els.note.hidden = false;
      els.note.textContent = `Forecasts are available for 1 March to 30 June of ${meta.years[0]}-${meta.years[meta.years.length - 1]}. Showing the nearest date, ${shortDate(issueDate)} ${issueDate.slice(0, 4)}.`;
    } else {
      els.note.hidden = true;
    }
    els.date.value = issueDate;
    syncUrl(regionName, modelName, issueDate);

    const k = data.origins.indexOf(origin);
    const q = data.q[k].map((row) => row.map(t10)); // [quantile][lead]
    const [pHot, pHw, pSev] = data.p[k];
    const H = meta.horizon;
    const tgt = Array.from({ length: H }, (_, i) => origin + 1 + i);
    const levels = tgt.map((_, i) => level(pHot[i], pHw[i], pSev[i], mod.thresholds));
    const worst = Math.max(...levels);
    const obsFuture = tgt.map((t) => t10(reg.tmax[t]));

    // banner and stats
    els.banner.className = `banner bd-${worst}`;
    els.banner.innerHTML = `<b class="lvl-${worst}">${LEVELS[worst].name}</b> for ${regionName}, next ${H} days: ${LEVELS[worst].text}. Highest heatwave probability ${Math.max(...pHw)}%.`;
    els.level.innerHTML = `<span class="lvl-${worst}">${LEVELS[worst].name}</span>`;
    els.hw.textContent = `${Math.max(...pHw)}%`;
    els.hot.textContent = `${Math.max(...pHot)}%`;

    // table
    els.tbody.innerHTML = tgt.map((t, i) => `
      <tr>
        <td>${longDate(meta.days[t])}</td>
        <td class="num">${fmt1(q[2][i])}</td>
        <td class="num">${fmt1(q[0][i])} to ${fmt1(q[4][i])}</td>
        <td class="num">${fmt1(t10(reg.thr_hw[t]))}</td>
        <td class="num">${pHot[i]}%</td>
        <td class="num">${pHw[i]}%</td>
        <td class="num">${pSev[i]}%</td>
        <td><span class="lv lvl-${levels[i]}">${LEVELS[levels[i]].name}</span></td>
        <td class="num">${fmt1(obsFuture[i])}</td>
      </tr>`).join("");

    els.rule.textContent = reg.type === "coastal"
      ? `${regionName} uses the IMD coastal rule: Tmax at least 37 °C and at least 4.5 °C above normal.`
      : `${regionName} uses the IMD plains rule: Tmax at least 40 °C and at least 4.5 °C above normal, or at least 45 °C.`;

    drawChart(reg, origin, tgt, q, pHot, pHw);
  }

  function drawChart(reg, origin, tgt, q, pHot, pHw) {
    const W = 900, Hh = 340, m = { l: 46, r: 14, t: 12, b: 30 };
    const hist = Array.from({ length: HISTORY + 1 }, (_, i) => origin - HISTORY + i);
    const all = hist.concat(tgt);
    const x0 = all[0], x1 = all[all.length - 1];
    const vals = [];
    hist.forEach((t) => vals.push(t10(reg.tmax[t])));
    tgt.forEach((t, i) => { vals.push(q[0][i], q[4][i], t10(reg.thr_hw[t]), t10(reg.thr_hot[t]), t10(reg.tmax[t])); });
    const clean = vals.filter((v) => v !== null && Number.isFinite(v));
    let yMin = Math.floor(Math.min(...clean) - 0.5), yMax = Math.ceil(Math.max(...clean) + 0.5);
    const X = (t) => m.l + ((t - x0) / (x1 - x0)) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - yMin) / (yMax - yMin)) * (Hh - m.t - m.b);

    const step = yMax - yMin > 10 ? 2 : 1;
    let svg = `<svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Forecast chart">`;
    for (let v = Math.ceil(yMin / step) * step; v <= yMax; v += step) {
      svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)" stroke-width="1"/>`;
      svg += `<text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end" font-size="12" fill="var(--ink-2)">${v}</text>`;
    }
    svg += `<text x="12" y="${m.t + (Hh - m.t - m.b) / 2}" transform="rotate(-90 12 ${m.t + (Hh - m.t - m.b) / 2})" text-anchor="middle" font-size="12" fill="var(--ink-2)">Tmax (°C)</text>`;
    all.forEach((t, i) => {
      if ((t - x0) % 4 === 0) {
        svg += `<text x="${X(t)}" y="${Hh - 8}" text-anchor="middle" font-size="12" fill="var(--ink-2)">${shortDate(meta.days[t])}</text>`;
      }
    });
    svg += `<line x1="${X(origin)}" x2="${X(origin)}" y1="${m.t}" y2="${Hh - m.b}" stroke="var(--ink-3)" stroke-dasharray="2 3"/>`;

    const band = (lo, hi, fill) => {
      const top = tgt.map((t, i) => `${X(t)},${Y(q[hi][i])}`);
      const bot = tgt.map((t, i) => `${X(t)},${Y(q[lo][i])}`).reverse();
      return `<polygon points="${top.concat(bot).join(" ")}" fill="${fill}"/>`;
    };
    svg += band(0, 4, "var(--fc-80)");
    svg += band(1, 3, "var(--fc-50)");

    const path = (pts) => pts.filter((p) => p[1] !== null).map((p, i) => `${i ? "L" : "M"}${X(p[0])},${Y(p[1])}`).join("");
    svg += `<path d="${path(tgt.map((t) => [t, t10(reg.thr_hot[t])]))}" fill="none" stroke="var(--hot)" stroke-width="1.6" stroke-dasharray="2 3"/>`;
    svg += `<path d="${path(tgt.map((t) => [t, t10(reg.thr_hw[t])]))}" fill="none" stroke="var(--thr)" stroke-width="2" stroke-dasharray="6 4"/>`;
    svg += `<path d="${path(hist.map((t) => [t, t10(reg.tmax[t])]))}" fill="none" stroke="var(--obs)" stroke-width="2"/>`;
    svg += `<path d="${path(tgt.map((t, i) => [t, q[2][i]]))}" fill="none" stroke="var(--fc)" stroke-width="2.5"/>`;
    tgt.forEach((t) => {
      const v = t10(reg.tmax[t]);
      if (v !== null) svg += `<circle cx="${X(t)}" cy="${Y(v)}" r="4.5" fill="var(--ink)" stroke="var(--card)" stroke-width="2"/>`;
    });
    svg += `<line id="xhair" x1="0" x2="0" y1="${m.t}" y2="${Hh - m.b}" stroke="var(--ink-3)" stroke-width="1" visibility="hidden"/>`;
    svg += `<rect id="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${Hh - m.t - m.b}" fill="transparent"/>`;
    svg += `</svg>`;
    els.chart.innerHTML = svg;

    els.legend.innerHTML = [
      `<span><i class="sw" style="border-color:var(--obs)"></i>Observed Tmax</span>`,
      `<span><i class="sw" style="border-color:var(--fc);border-top-width:3px"></i>Median forecast</span>`,
      `<span><i class="sw band" style="background:var(--fc-50)"></i>50% range</span>`,
      `<span><i class="sw band" style="background:var(--fc-80)"></i>80% range</span>`,
      `<span><i class="sw dot" style="background:var(--ink)"></i>Observed (verification)</span>`,
      `<span><i class="sw" style="border-color:var(--thr);border-top-style:dashed"></i>IMD heatwave threshold</span>`,
      `<span><i class="sw" style="border-color:var(--hot);border-top-style:dotted"></i>Hot day (90th percentile)</span>`,
    ].join("");

    // hover: nearest day, crosshair and tooltip
    const svgEl = els.chart.querySelector("svg");
    const hit = svgEl.querySelector("#hit");
    const xhair = svgEl.querySelector("#xhair");
    const card = els.chart.parentElement;
    const show = (evt) => {
      const pt = svgEl.createSVGPoint();
      pt.x = evt.clientX; pt.y = evt.clientY;
      const loc = pt.matrixTransform(svgEl.getScreenCTM().inverse());
      const t = Math.round(x0 + ((loc.x - m.l) / (W - m.l - m.r)) * (x1 - x0));
      if (t < x0 || t > x1) return;
      xhair.setAttribute("x1", X(t)); xhair.setAttribute("x2", X(t)); xhair.setAttribute("visibility", "visible");
      const i = tgt.indexOf(t);
      const obs = t10(reg.tmax[t]);
      let rows = `<div class="r">Observed <b>${obs === null ? "n/a" : fmt1(obs) + " °C"}</b></div>`;
      if (i >= 0) {
        rows += `<div class="r">Median <b>${fmt1(q[2][i])} °C</b></div>`;
        rows += `<div class="r">80% range <b>${fmt1(q[0][i])} to ${fmt1(q[4][i])}</b></div>`;
        rows += `<div class="r">HW threshold <b>${fmt1(t10(reg.thr_hw[t]))} °C</b></div>`;
        rows += `<div class="r">P(hot day) <b>${pHot[i]}%</b></div>`;
        rows += `<div class="r">P(heatwave) <b>${pHw[i]}%</b></div>`;
      }
      els.tooltip.innerHTML = `<div class="t">${longDate(meta.days[t])} ${meta.days[t].slice(0, 4)}${i >= 0 ? `, lead ${i + 1} day${i ? "s" : ""}` : ""}</div>${rows}`;
      els.tooltip.hidden = false;
      const cardBox = card.getBoundingClientRect();
      let left = evt.clientX - cardBox.left + 14;
      if (left + els.tooltip.offsetWidth > cardBox.width - 6) left = evt.clientX - cardBox.left - els.tooltip.offsetWidth - 14;
      els.tooltip.style.left = `${left}px`;
      els.tooltip.style.top = `${Math.max(6, evt.clientY - cardBox.top - 20)}px`;
    };
    hit.addEventListener("mousemove", show);
    hit.addEventListener("touchstart", (e) => show(e.touches[0]), { passive: true });
    hit.addEventListener("mouseleave", () => { els.tooltip.hidden = true; xhair.setAttribute("visibility", "hidden"); });
  }

  async function init() {
    meta = await loadJSON("data/meta.json");
    els.region.innerHTML = meta.regions.map((r) => `<option>${r.name}</option>`).join("");
    els.model.innerHTML = meta.models.map((m) => `<option>${m.name}</option>`).join("");
    els.date.min = `${meta.years[0]}-03-01`;
    els.date.max = `${meta.years[meta.years.length - 1]}-06-30`;

    const p = new URLSearchParams(location.search);
    if (p.get("region") && meta.regions.some((r) => r.name === p.get("region"))) els.region.value = p.get("region");
    if (p.get("model") && meta.models.some((m) => m.name === p.get("model"))) els.model.value = p.get("model");
    els.date.value = p.get("date") || `${meta.years[meta.years.length - 1]}-05-10`;

    [els.region, els.model, els.date].forEach((el) => el.addEventListener("change", render));
    document.querySelectorAll(".examples button").forEach((b) => b.addEventListener("click", () => {
      els.region.value = b.dataset.r;
      els.date.value = b.dataset.d;
      render();
    }));
    render();
  }

  init().catch((e) => {
    els.banner.textContent = `Could not load forecast data: ${e.message}`;
  });
})();
