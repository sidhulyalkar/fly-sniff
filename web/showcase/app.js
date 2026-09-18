(() => {
  "use strict";

  const qs = (id) => document.getElementById(id);
  const arenaCanvas = qs("arenaCanvas");
  const connectomeCanvas = qs("connectomeCanvas");
  const arenaCtx = arenaCanvas.getContext("2d");
  const connectomeCtx = connectomeCanvas.getContext("2d");

  const state = {
    data: null,
    playing: true,
    frameIndex: 0,
    selected: null,
    raf: 0,
    lastTs: 0,
    accumulator: 0,
    somaContext: null,
    skeletons: null,
    viewYaw: -0.55,
    viewPitch: 0.24,
    dragging: false,
    dragX: 0,
    dragY: 0,
  };

  const palette = ["#47c7f3", "#ff6f91", "#f5bf42", "#9e8cff", "#9bd650", "#f06464"];

  function setBadge(el, text, kind) {
    el.textContent = text;
    el.className = "badge" + (kind ? " " + kind : "");
  }

  function fitCanvas(canvas, ctx) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { width: rect.width, height: rect.height };
  }

  function arenaPoint(x, y, dims) {
    const a = state.data.arena;
    const pad = 28;
    return {
      x: pad + (x / a.width) * Math.max(1, dims.width - 2 * pad),
      y: dims.height - pad - (y / a.height) * Math.max(1, dims.height - 2 * pad),
    };
  }

  function drawArena() {
    if (!state.data) return;
    const dims = fitCanvas(arenaCanvas, arenaCtx);
    arenaCtx.clearRect(0, 0, dims.width, dims.height);

    const grad = arenaCtx.createLinearGradient(0, 0, dims.width, dims.height);
    grad.addColorStop(0, "#071624");
    grad.addColorStop(1, "#050d16");
    arenaCtx.fillStyle = grad;
    arenaCtx.fillRect(0, 0, dims.width, dims.height);

    arenaCtx.strokeStyle = "rgba(78,110,139,.13)";
    arenaCtx.lineWidth = 1;
    for (let i = 1; i < 6; i++) {
      const x = (dims.width / 6) * i;
      arenaCtx.beginPath();
      arenaCtx.moveTo(x, 0);
      arenaCtx.lineTo(x, dims.height);
      arenaCtx.stroke();
    }
    for (let i = 1; i < 4; i++) {
      const y = (dims.height / 4) * i;
      arenaCtx.beginPath();
      arenaCtx.moveTo(0, y);
      arenaCtx.lineTo(dims.width, y);
      arenaCtx.stroke();
    }

    const frame = state.data.frames[state.frameIndex];
    if (!frame) return;

    if (qs("showPlume").checked) {
      for (const puff of frame.plume || []) {
        const p = arenaPoint(puff[0], puff[1], dims);
        const radius = Math.max(2, puff[2] * 14);
        const g = arenaCtx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius * 2.2);
        g.addColorStop(0, "rgba(139,207,76,.24)");
        g.addColorStop(1, "rgba(139,207,76,0)");
        arenaCtx.fillStyle = g;
        arenaCtx.beginPath();
        arenaCtx.arc(p.x, p.y, radius * 2.2, 0, Math.PI * 2);
        arenaCtx.fill();
      }
    }

    if (qs("showSource").checked) {
      const s = arenaPoint(state.data.arena.source_x, state.data.arena.source_y, dims);
      arenaCtx.strokeStyle = "#9bd650";
      arenaCtx.lineWidth = 2;
      arenaCtx.setLineDash([5, 4]);
      arenaCtx.beginPath();
      arenaCtx.arc(s.x, s.y, 11, 0, Math.PI * 2);
      arenaCtx.stroke();
      arenaCtx.setLineDash([]);
      arenaCtx.fillStyle = "#9bd650";
      arenaCtx.font = "600 10px system-ui";
      arenaCtx.fillText("VIEWER-ONLY SOURCE", s.x + 16, s.y + 4);
    }

    state.data.conditions.forEach((condition, index) => {
      const agent = frame.agents[condition.key];
      if (!agent) return;
      const color = palette[index % palette.length];

      arenaCtx.strokeStyle = color + "88";
      arenaCtx.lineWidth = 1.6;
      arenaCtx.beginPath();
      let started = false;
      for (let i = 0; i <= state.frameIndex; i++) {
        const a = state.data.frames[i]?.agents?.[condition.key];
        if (!a) continue;
        const p = arenaPoint(a.x, a.y, dims);
        if (!started) {
          arenaCtx.moveTo(p.x, p.y);
          started = true;
        } else {
          arenaCtx.lineTo(p.x, p.y);
        }
      }
      arenaCtx.stroke();

      const p = arenaPoint(agent.x, agent.y, dims);
      arenaCtx.save();
      arenaCtx.translate(p.x, p.y);
      arenaCtx.rotate(-agent.heading);
      arenaCtx.fillStyle = color;
      arenaCtx.shadowColor = color;
      arenaCtx.shadowBlur = condition.key === state.selected ? 15 : 4;
      arenaCtx.beginPath();
      arenaCtx.moveTo(10, 0);
      arenaCtx.lineTo(-6, -5);
      arenaCtx.lineTo(-3, 0);
      arenaCtx.lineTo(-6, 5);
      arenaCtx.closePath();
      arenaCtx.fill();
      arenaCtx.restore();

      if (condition.key === state.selected) {
        arenaCtx.strokeStyle = color;
        arenaCtx.lineWidth = 1;
        arenaCtx.beginPath();
        arenaCtx.arc(p.x, p.y, 15, 0, Math.PI * 2);
        arenaCtx.stroke();
      }
    });

    arenaCtx.fillStyle = "rgba(190,210,228,.65)";
    arenaCtx.font = "11px ui-monospace, SFMono-Regular, monospace";
    arenaCtx.fillText("wind →", dims.width - 75, 22);
  }

  function hashUnit(id) {
    let x = Number(id) || 1;
    x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
    x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
    return (x ^ (x >>> 16)) >>> 0;
  }

  function topologyPosition(node, index, total, dims) {
    const h = hashUnit(node.body_id);
    const angle = ((h % 100000) / 100000) * Math.PI * 2;
    const ring = 0.20 + (((h >>> 8) % 1000) / 1000) * 0.72;
    const rx = dims.width * 0.42 * ring;
    const ry = dims.height * 0.38 * ring;
    return {
      x: dims.width / 2 + Math.cos(angle) * rx,
      y: dims.height / 2 + Math.sin(angle) * ry,
    };
  }

  function roleStrength(roles, agent) {
    let value = 0.08;
    for (const role of roles || []) {
      if (role === "odor_left") value = Math.max(value, agent?.left_odor || 0);
      if (role === "odor_right") value = Math.max(value, agent?.right_odor || 0);
      if (role === "steer_left") value = Math.max(value, agent?.dn_left || 0);
      if (role === "steer_right") value = Math.max(value, agent?.dn_right || 0);
    }
    return Math.min(1, value);
  }

  function anatomyBounds() {
    if (state.somaContext?.bounds?.min && state.somaContext?.bounds?.max) {
      return { min: state.somaContext.bounds.min, max: state.somaContext.bounds.max };
    }
    const all = [];
    for (const neuron of state.skeletons?.neurons || []) {
      for (const seg of neuron.segments || []) {
        all.push([seg[0], seg[1], seg[2]], [seg[3], seg[4], seg[5]]);
      }
    }
    if (!all.length) return null;
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const p of all) {
      for (let k = 0; k < 3; k++) {
        min[k] = Math.min(min[k], Number(p[k]));
        max[k] = Math.max(max[k], Number(p[k]));
      }
    }
    return { min, max };
  }

  function projectXYZ(x, y, z, dims, bounds) {
    const center = bounds.min.map((v, i) => (Number(v) + Number(bounds.max[i])) / 2);
    const span = bounds.min.map((v, i) => Math.max(1, Number(bounds.max[i]) - Number(v)));
    const scale = Math.min(dims.width, dims.height) * 0.78 / Math.max(...span);

    let px = (Number(x) - center[0]) * scale;
    let py = (Number(y) - center[1]) * scale;
    let pz = (Number(z) - center[2]) * scale;

    const cy = Math.cos(state.viewYaw);
    const sy = Math.sin(state.viewYaw);
    const x1 = cy * px + sy * pz;
    const z1 = -sy * px + cy * pz;
    const cp = Math.cos(state.viewPitch);
    const sp = Math.sin(state.viewPitch);
    const y1 = cp * py - sp * z1;
    const z2 = sp * py + cp * z1;

    return {
      x: dims.width / 2 + x1,
      y: dims.height / 2 - y1,
      depth: z2,
    };
  }

  function skeletonColor(roles, agent, bodyId) {
    const roleValue = roleStrength(roles, agent);
    const rawActivity = Math.abs(Number(agent?.node_activity?.[String(bodyId)] || 0));
    const activity = Math.max(0, Math.min(1, rawActivity));
    const strength = Math.max(roleValue, activity);
    if ((roles || []).some((r) => r.startsWith("odor_"))) {
      return `rgba(71,199,243,${0.20 + 0.76 * strength})`;
    }
    if ((roles || []).some((r) => r.startsWith("steer_"))) {
      return `rgba(245,191,66,${0.20 + 0.76 * strength})`;
    }
    return `rgba(158,140,255,${0.12 + 0.78 * strength})`;
  }

  function drawMeasuredAnatomy(dims, selectedAgent) {
    const bounds = anatomyBounds();
    if (!bounds) return false;

    if (state.somaContext?.points?.length) {
      const points = state.somaContext.points;
      for (let i = 0; i < points.length; i++) {
        const row = points[i];
        const p = projectXYZ(row.x, row.y, row.z, dims, bounds);
        const alpha = 0.10 + Math.max(-0.04, Math.min(0.05, p.depth * 0.000001));
        connectomeCtx.fillStyle = `rgba(103,137,165,${alpha})`;
        connectomeCtx.fillRect(p.x, p.y, 1.2, 1.2);
      }
    }

    if (state.skeletons?.neurons?.length) {
      connectomeCtx.lineWidth = 1.0;
      for (const neuron of state.skeletons.neurons) {
        connectomeCtx.strokeStyle = skeletonColor(
          neuron.roles || [],
          selectedAgent,
          neuron.body_id
        );
        connectomeCtx.beginPath();
        for (const seg of neuron.segments || []) {
          const a = projectXYZ(seg[0], seg[1], seg[2], dims, bounds);
          const b = projectXYZ(seg[3], seg[4], seg[5], dims, bounds);
          connectomeCtx.moveTo(a.x, a.y);
          connectomeCtx.lineTo(b.x, b.y);
        }
        connectomeCtx.stroke();
      }
    }

    connectomeCtx.fillStyle = "rgba(207,229,244,.72)";
    connectomeCtx.font = "10px ui-monospace, SFMono-Regular, monospace";
    const label = state.skeletons?.neurons?.length
      ? "measured soma context + selected SWC skeletons • drag to rotate"
      : "measured somaLocation context • points are somata, not neurites • drag to rotate";
    connectomeCtx.fillText(label, 12, dims.height - 14);
    return true;
  }

  function drawTopology(dims, selectedAgent) {
    const c = state.data.connectome || {};
    const nodes = c.nodes || [];
    const positions = new Map();
    nodes.forEach((node, i) => {
      positions.set(String(node.body_id), topologyPosition(node, i, nodes.length, dims));
    });

    connectomeCtx.strokeStyle = "rgba(89,125,154,.10)";
    connectomeCtx.lineWidth = 0.7;
    for (const edge of c.edges || []) {
      const a = positions.get(String(edge.source));
      const b = positions.get(String(edge.target));
      if (!a || !b) continue;
      connectomeCtx.beginPath();
      connectomeCtx.moveTo(a.x, a.y);
      connectomeCtx.lineTo(b.x, b.y);
      connectomeCtx.stroke();
    }

    for (const node of nodes) {
      const p = positions.get(String(node.body_id));
      if (!p) continue;
      const roleValue = roleStrength(node.roles, selectedAgent);
      const activity = Math.min(
        1,
        Math.abs(Number(selectedAgent?.node_activity?.[String(node.body_id)] || 0))
      );
      const strength = Math.max(roleValue, activity);
      const isRole = (node.roles || []).length > 0;
      connectomeCtx.fillStyle = isRole || activity > 0
        ? `rgba(71,199,243,${0.20 + strength * 0.75})`
        : "rgba(116,145,171,.28)";
      connectomeCtx.beginPath();
      connectomeCtx.arc(p.x, p.y, isRole ? 3 + strength * 3 : 1.5, 0, Math.PI * 2);
      connectomeCtx.fill();
    }

    connectomeCtx.fillStyle = "rgba(207,229,244,.68)";
    connectomeCtx.font = "10px ui-monospace, SFMono-Regular, monospace";
    connectomeCtx.fillText(
      "graph topology • deterministic layout • not anatomical XYZ",
      12,
      dims.height - 14
    );
  }

  function drawConnectome() {
    if (!state.data) return;
    const dims = fitCanvas(connectomeCanvas, connectomeCtx);
    connectomeCtx.clearRect(0, 0, dims.width, dims.height);

    const frame = state.data.frames[state.frameIndex];
    const selectedAgent = frame?.agents?.[state.selected] || null;
    const empty = qs("connectomeEmpty");

    if (drawMeasuredAnatomy(dims, selectedAgent)) {
      empty.hidden = true;
      return;
    }

    const c = state.data.connectome || {};
    if (!c.available) {
      empty.hidden = false;
      empty.textContent = c.claim_boundary || "No connectome asset is loaded.";
      return;
    }

    empty.hidden = true;
    drawTopology(dims, selectedAgent);
  }

  function updateTelemetry() {
    if (!state.data) return;
    const frame = state.data.frames[state.frameIndex];
    const condition = state.data.conditions.find((c) => c.key === state.selected);
    const agent = frame?.agents?.[state.selected];
    if (!condition || !agent) return;

    qs("selectedLabel").textContent = condition.label;
    qs("leftOdorValue").textContent = agent.left_odor.toFixed(2);
    qs("rightOdorValue").textContent = agent.right_odor.toFixed(2);
    qs("turnValue").textContent = agent.turn.toFixed(2);
    qs("readoutValue").textContent = `${agent.dn_left.toFixed(2)} / ${agent.dn_right.toFixed(2)}`;
    qs("leftOdorBar").style.width = `${Math.max(0, Math.min(100, agent.left_odor * 100))}%`;
    qs("rightOdorBar").style.width = `${Math.max(0, Math.min(100, agent.right_odor * 100))}%`;

    const turn = Math.max(-1, Math.min(1, agent.turn));
    const turnBar = qs("turnBar");
    turnBar.style.left = turn >= 0 ? "50%" : `${50 + turn * 50}%`;
    turnBar.style.width = `${Math.abs(turn) * 50}%`;
    turnBar.style.background = turn >= 0 ? "#9e8cff" : "#f5bf42";

    qs("dnLeftBar").style.transform = `scaleX(${Math.max(0, Math.min(1, agent.dn_left))})`;
    qs("dnRightBar").style.transform = `scaleX(${Math.max(0, Math.min(1, agent.dn_right))})`;

    qs("interventionLabel").textContent =
      condition.intervention === "none" ? "No intervention" : condition.intervention;
    qs("conditionDescription").textContent = condition.description || "";
    qs("timeOutput").textContent = `${Number(frame.t).toFixed(1)} s`;
    qs("timeline").value = String(
      state.data.frames.length <= 1
        ? 0
        : Math.round((state.frameIndex / (state.data.frames.length - 1)) * 1000)
    );
  }

  function renderAll() {
    drawArena();
    drawConnectome();
    updateTelemetry();
  }

  function setupLegend() {
    const legend = qs("arenaLegend");
    legend.innerHTML = "";
    state.data.conditions.forEach((condition, index) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "legend-item";
      item.dataset.key = condition.key;
      item.innerHTML = `<span class="legend-dot" style="background:${palette[index % palette.length]}"></span><span></span>`;
      item.querySelector("span:last-child").textContent = condition.label;
      item.addEventListener("click", () => selectCondition(condition.key));
      legend.appendChild(item);
    });
  }

  function selectCondition(key) {
    state.selected = key;
    qs("conditionSelect").value = key;
    renderAll();
  }

  function setupConditionSelect() {
    const select = qs("conditionSelect");
    select.innerHTML = "";
    state.data.conditions.forEach((condition) => {
      const option = document.createElement("option");
      option.value = condition.key;
      option.textContent = condition.label;
      select.appendChild(option);
    });
    select.addEventListener("change", () => selectCondition(select.value));
  }

  function configureEvidence() {
    const qualified = Boolean(state.data.claim_allowed);
    setBadge(
      qs("modeBadge"),
      qualified ? "QUALIFIED MODELED CIRCUIT" : "DEVELOPMENT PROXY",
      qualified ? "badge-qualified" : "badge-dev"
    );
    setBadge(
      qs("claimBadge"),
      qualified ? "odor-navigation qualification loaded" : "no MaleCNS navigation claim",
      qualified ? "badge-qualified" : "badge-warn"
    );

    const connectome = state.data.connectome || {};
    if (state.somaContext || state.skeletons) {
      qs("connectomeTitle").textContent = state.skeletons
        ? "Measured MaleCNS context + selected circuit morphology"
        : "Measured MaleCNS soma context";
      setBadge(
        qs("geometryBadge"),
        state.skeletons ? "MEASURED SOMA + SWC" : "MEASURED SOMA XYZ",
        "badge-qualified"
      );
      const pieces = [];
      if (state.somaContext?.point_meaning) pieces.push(state.somaContext.point_meaning);
      if (state.skeletons?.claim_boundary) pieces.push(state.skeletons.claim_boundary);
      qs("connectomeBoundary").textContent = pieces.join(" ");
    } else {
      qs("connectomeTitle").textContent = connectome.label || "Connectome status";
      setBadge(
        qs("geometryBadge"),
        connectome.geometry_kind === "topology_only"
          ? "TOPOLOGY • NOT MORPHOLOGY"
          : "NO CONNECTOME ASSET",
        "badge-warn"
      );
      qs("connectomeBoundary").textContent = connectome.claim_boundary || "";
    }

    qs("globalBoundary").textContent = state.data.claim_boundary || "";
    qs("schemaLabel").textContent = state.data.schema || "unknown schema";
  }

  function animationLoop(ts) {
    if (!state.data) return;
    if (!state.lastTs) state.lastTs = ts;
    const dt = Math.max(0, Math.min(100, ts - state.lastTs));
    state.lastTs = ts;

    if (state.playing && state.data.frames.length > 1) {
      state.accumulator += dt;
      const frameMs = 1000 / Math.max(1, state.data.sample_hz || 10);
      if (state.accumulator >= frameMs) {
        const steps = Math.floor(state.accumulator / frameMs);
        state.accumulator -= steps * frameMs;
        state.frameIndex = (state.frameIndex + steps) % state.data.frames.length;
        renderAll();
      }
    }
    state.raf = requestAnimationFrame(animationLoop);
  }

  function wireControls() {
    qs("playButton").addEventListener("click", () => {
      state.playing = !state.playing;
      qs("playButton").textContent = state.playing ? "Ⅱ" : "▶";
      qs("playButton").setAttribute("aria-label", state.playing ? "Pause replay" : "Play replay");
    });

    qs("timeline").addEventListener("input", () => {
      const fraction = Number(qs("timeline").value) / 1000;
      state.frameIndex = Math.max(
        0,
        Math.min(state.data.frames.length - 1, Math.round(fraction * (state.data.frames.length - 1)))
      );
      state.accumulator = 0;
      renderAll();
    });

    qs("showPlume").addEventListener("change", drawArena);
    qs("showSource").addEventListener("change", drawArena);

    connectomeCanvas.addEventListener("pointerdown", (event) => {
      state.dragging = true;
      state.dragX = event.clientX;
      state.dragY = event.clientY;
      connectomeCanvas.setPointerCapture?.(event.pointerId);
    });
    connectomeCanvas.addEventListener("pointermove", (event) => {
      if (!state.dragging || (!state.somaContext && !state.skeletons)) return;
      const dx = event.clientX - state.dragX;
      const dy = event.clientY - state.dragY;
      state.dragX = event.clientX;
      state.dragY = event.clientY;
      state.viewYaw += dx * 0.008;
      state.viewPitch = Math.max(-1.15, Math.min(1.15, state.viewPitch + dy * 0.008));
      drawConnectome();
    });
    const endDrag = () => { state.dragging = false; };
    connectomeCanvas.addEventListener("pointerup", endDrag);
    connectomeCanvas.addEventListener("pointercancel", endDrag);

    window.addEventListener("resize", renderAll);
  }

  async function loadOptionalJSON(url) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) return null;
      return await response.json();
    } catch (_) {
      return null;
    }
  }

  async function init() {
    try {
      const response = await fetch("./data/showcase.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`showcase.json returned ${response.status}`);
      state.data = await response.json();
      const [somaContext, skeletons] = await Promise.all([
        loadOptionalJSON("./data/connectome-soma.json"),
        loadOptionalJSON("./data/selected-skeletons.json"),
      ]);
      state.somaContext = somaContext;
      state.skeletons = skeletons;

      if (!Array.isArray(state.data.frames) || state.data.frames.length === 0) {
        throw new Error("showcase.json has no replay frames");
      }
      if (!Array.isArray(state.data.conditions) || state.data.conditions.length === 0) {
        throw new Error("showcase.json has no conditions");
      }

      state.selected = state.data.conditions[0].key;
      qs("timeline").max = "1000";
      setupLegend();
      setupConditionSelect();
      configureEvidence();
      wireControls();
      renderAll();

      qs("o002Image").addEventListener("error", () => {
        qs("o002Image").hidden = true;
        qs("o002Fallback").hidden = false;
        qs("o002Link").hidden = true;
      }, { once: true });

      state.raf = requestAnimationFrame(animationLoop);
    } catch (error) {
      console.error(error);
      setBadge(qs("modeBadge"), "DATA LOAD FAILED", "badge-warn");
      setBadge(qs("claimBadge"), "run site builder first", "badge-warn");
      qs("globalBoundary").textContent =
        "No replay data loaded. Run scripts/build_live_showcase_mac.sh and serve the generated directory over HTTP.";
      qs("connectomeEmpty").hidden = false;
      qs("connectomeEmpty").textContent =
        "The frontend shell is ready, but ./data/showcase.json has not been generated.";
    }
  }

  init();

  window.addEventListener("beforeunload", () => {
    if (state.raf) cancelAnimationFrame(state.raf);
  });
})();