import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./style.css";

const STREAM_URL = "/data/replay-stream-v1.json";
const ASSET_URL = "/data/route-skeletons-v1.json";

const worldCanvas = document.querySelector("#world-canvas");
const brainHost = document.querySelector("#brain-canvas");
const behaviorClock = document.querySelector("#behavior-clock");
const brainStatus = document.querySelector("#brain-status");
const evidenceNote = document.querySelector("#evidence-note");
const odorLeft = document.querySelector("#odor-left");
const odorRight = document.querySelector("#odor-right");
const odorLeftMeter = document.querySelector("#odor-left-meter");
const odorRightMeter = document.querySelector("#odor-right-meter");
const windReadout = document.querySelector("#wind-readout");
const windGlyph = document.querySelector("#wind-glyph");
const turnCommand = document.querySelector("#turn-command");
const actionLabel = document.querySelector("#action-label");
const railAction = document.querySelector("#rail-action");
const timelineWarning = document.querySelector("#timeline-warning");
const scrubber = document.querySelector("#scrubber");
const playButton = document.querySelector("#play-button");
const playIcon = document.querySelector("#play-icon");
const durationLabel = document.querySelector("#duration-label");
const plumeIntegrity = document.querySelector("#plume-integrity");
const mechanismTitle = document.querySelector("#mechanism-title");
const mechanismPass = document.querySelector("#mechanism-pass");
const mechanismCopy = document.querySelector("#mechanism-copy");
const qualificationPill = document.querySelector("#qualification-pill");
const phaseCurve = document.querySelector("#phase-curve");
const neuronHover = document.querySelector("#neuron-hover");
const filterButtons = [...document.querySelectorAll(".filter-chip")];

let stream = null;
let anatomy = null;
let behaviorIndex = 0;
let playing = true;
let lastTick = performance.now();
let activeFilter = "ALL";

const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, ratio };
}

function worldToCanvas(x, y, world, width, height, pad = 38) {
  const sx = (width - 2 * pad) / world.width;
  const sy = (height - 2 * pad) / world.height;
  return [pad + x * sx, height - pad - y * sy];
}

function drawArenaGrid(ctx, width, height, ratio) {
  ctx.save();
  ctx.strokeStyle = "rgba(125, 151, 177, .055)";
  ctx.lineWidth = ratio;
  for (let i = 1; i < 5; i += 1) {
    ctx.beginPath();
    ctx.moveTo((width * i) / 5, 0);
    ctx.lineTo((width * i) / 5, height);
    ctx.stroke();
  }
  for (let i = 1; i < 4; i += 1) {
    ctx.beginPath();
    ctx.moveTo(0, (height * i) / 4);
    ctx.lineTo(width, (height * i) / 4);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPlume(ctx, frame, world, width, height, ratio) {
  const complete = frame.plume_snapshot_complete === true;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const [x, y, sigma] of frame.plume_components) {
    const [px, py] = worldToCanvas(x, y, world, width, height);
    const radius = Math.max(1.2 * ratio, sigma * 8.2 * ratio);
    const grad = ctx.createRadialGradient(px, py, 0, px, py, radius * 3.3);
    grad.addColorStop(0, complete ? "rgba(163,230,53,.13)" : "rgba(163,230,53,.085)");
    grad.addColorStop(.28, complete ? "rgba(132,204,22,.075)" : "rgba(132,204,22,.052)");
    grad.addColorStop(1, "rgba(77,124,15,0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(px, py, radius * 3.3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawTrajectory(ctx, frames, agentIndex, world, width, height, ratio, color) {
  const maxTrail = 180;
  const start = Math.max(0, frames.length - maxTrail);
  if (frames.length - start < 2) return;
  for (let i = start + 1; i < frames.length; i += 1) {
    const previous = frames[i - 1]?.agents?.[agentIndex];
    const current = frames[i]?.agents?.[agentIndex];
    if (!previous || !current) continue;
    const [x0, y0] = worldToCanvas(previous.position[0], previous.position[1], world, width, height);
    const [x1, y1] = worldToCanvas(current.position[0], current.position[1], world, width, height);
    const age = (i - start) / Math.max(frames.length - start, 1);
    ctx.strokeStyle = color.replace("ALPHA", String(.04 + .32 * age));
    ctx.lineWidth = (1.0 + 1.5 * age) * ratio;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }
}

function antennaPoints(agent, world) {
  const half = .5 * Number(world.antenna_separation || .18);
  const s = Math.sin(agent.heading_rad);
  const c = Math.cos(agent.heading_rad);
  return {
    left: [agent.position[0] - s * half, agent.position[1] + c * half],
    right: [agent.position[0] + s * half, agent.position[1] - c * half],
  };
}

function drawWindVector(ctx, agent, world, width, height, ratio) {
  const [bx, by] = agent.sensors.wind_body;
  const magnitude = Math.hypot(bx, by);
  if (magnitude < 1e-6) return;
  const worldAngle = agent.heading_rad + Math.atan2(by, bx);
  const [px, py] = worldToCanvas(agent.position[0], agent.position[1], world, width, height);
  const length = (22 + 18 * clamp(magnitude, 0, 1.5)) * ratio;
  const dx = Math.cos(worldAngle) * length;
  const dy = -Math.sin(worldAngle) * length;
  ctx.save();
  ctx.strokeStyle = "rgba(125,211,252,.46)";
  ctx.fillStyle = "rgba(125,211,252,.60)";
  ctx.lineWidth = 1.1 * ratio;
  ctx.setLineDash([3 * ratio, 4 * ratio]);
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(px + dx, py + dy);
  ctx.stroke();
  ctx.setLineDash([]);
  const angle = Math.atan2(dy, dx);
  ctx.beginPath();
  ctx.moveTo(px + dx, py + dy);
  ctx.lineTo(px + dx - 6 * ratio * Math.cos(angle - .45), py + dy - 6 * ratio * Math.sin(angle - .45));
  ctx.lineTo(px + dx - 6 * ratio * Math.cos(angle + .45), py + dy - 6 * ratio * Math.sin(angle + .45));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawAgent(ctx, agent, idx, world, width, height, ratio) {
  const [x, y] = worldToCanvas(agent.position[0], agent.position[1], world, width, height);
  const color = idx === 0 ? "#56d6ff" : "#fb7185";
  const heading = -agent.heading_rad;
  const scale = idx === 0 ? 1 : .88;

  if (idx === 0) {
    const antenna = antennaPoints(agent, world);
    for (const point of [antenna.left, antenna.right]) {
      const [ax, ay] = worldToCanvas(point[0], point[1], world, width, height);
      ctx.fillStyle = "rgba(103,232,249,.76)";
      ctx.beginPath();
      ctx.arc(ax, ay, 2.15 * ratio, 0, Math.PI * 2);
      ctx.fill();
    }
    drawWindVector(ctx, agent, world, width, height, ratio);
  }

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(heading);
  ctx.scale(scale, scale);
  ctx.shadowColor = color;
  ctx.shadowBlur = (idx === 0 ? 18 : 9) * ratio;

  ctx.fillStyle = idx === 0 ? "rgba(86,214,255,.20)" : "rgba(251,113,133,.13)";
  ctx.beginPath();
  ctx.ellipse(-3 * ratio, -5 * ratio, 8 * ratio, 4 * ratio, -.45, 0, Math.PI * 2);
  ctx.ellipse(-3 * ratio, 5 * ratio, 8 * ratio, 4 * ratio, .45, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.ellipse(0, 0, 9 * ratio, 4.2 * ratio, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(8 * ratio, 0, 4.2 * ratio, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = idx === 0 ? "rgba(207,250,254,.72)" : "rgba(254,205,211,.54)";
  ctx.lineWidth = 1.1 * ratio;
  ctx.beginPath();
  ctx.moveTo(11 * ratio, 0);
  ctx.lineTo(18 * ratio, 0);
  ctx.stroke();
  ctx.restore();
}

function drawSource(ctx, world, width, height, ratio) {
  if (!world.source_for_viewer_only) return;
  const [sx, sy] = worldToCanvas(
    world.source_for_viewer_only[0],
    world.source_for_viewer_only[1],
    world,
    width,
    height,
  );
  ctx.save();
  ctx.strokeStyle = "rgba(190,242,100,.58)";
  ctx.lineWidth = 1.2 * ratio;
  ctx.setLineDash([3 * ratio, 4 * ratio]);
  ctx.beginPath();
  ctx.arc(sx, sy, 9 * ratio, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(190,242,100,.56)";
  ctx.font = `${8 * ratio}px Inter`;
  ctx.fillText("VIEWER-ONLY SOURCE", sx + 14 * ratio, sy - 7 * ratio);
  ctx.restore();
}

function drawWorld(frame, world, frameIndex = behaviorIndex) {
  const ctx = worldCanvas.getContext("2d");
  const { width, height, ratio } = fitCanvas(worldCanvas);
  ctx.clearRect(0, 0, width, height);

  const background = ctx.createRadialGradient(width * .53, height * .47, 0, width * .53, height * .47, Math.max(width, height) * .72);
  background.addColorStop(0, "#0a1723");
  background.addColorStop(.62, "#07111b");
  background.addColorStop(1, "#050b12");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);

  drawArenaGrid(ctx, width, height, ratio);
  drawPlume(ctx, frame, world, width, height, ratio);

  const frames = stream?.timelines?.behavior?.frames || [];
  const history = frames.slice(0, frameIndex + 1);
  drawTrajectory(ctx, history, 0, world, width, height, ratio, "rgba(86,214,255,ALPHA)");
  drawTrajectory(ctx, history, 1, world, width, height, ratio, "rgba(251,113,133,ALPHA)");
  drawSource(ctx, world, width, height, ratio);
  frame.agents.forEach((agent, idx) => drawAgent(ctx, agent, idx, world, width, height, ratio));

  const primary = frame.agents[0];
  behaviorClock.textContent = `${frame.t_s.toFixed(2)} s`;
  odorLeft.textContent = primary.sensors.odor_left.toFixed(2);
  odorRight.textContent = primary.sensors.odor_right.toFixed(2);
  odorLeftMeter.style.width = `${100 * clamp(primary.sensors.odor_left, 0, 1)}%`;
  odorRightMeter.style.width = `${100 * clamp(primary.sensors.odor_right, 0, 1)}%`;

  const [windX, windY] = primary.sensors.wind_body;
  windReadout.textContent = `${windX.toFixed(2)} · ${windY.toFixed(2)}`;
  windGlyph.style.transform = `rotate(${-Math.atan2(windY, windX)}rad)`;

  const turn = primary.command.turn;
  turnCommand.textContent = turn.toFixed(2);
  const action = Math.abs(turn) < .08 ? "STRAIGHT" : turn > 0 ? "TURN LEFT" : "TURN RIGHT";
  actionLabel.textContent = action;
  railAction.textContent = action.toLowerCase();

  if (frame.plume_snapshot_metadata_available === false) {
    plumeIntegrity.textContent = "legacy plume · completeness unknown";
  } else if (frame.plume_snapshot_complete === true) {
    plumeIntegrity.textContent = "complete recorded plume snapshot";
  } else {
    plumeIntegrity.textContent = "sampled plume snapshot";
  }
}

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x070b12, .00145);
const camera = new THREE.PerspectiveCamera(36, 1, .01, 100000);
camera.position.set(0, 0, 1200);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x000000, 0);
brainHost.insertBefore(renderer.domElement, brainHost.firstChild);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = .075;
controls.autoRotate = true;
controls.autoRotateSpeed = .18;
controls.minDistance = 100;
controls.maxDistance = 5000;
const brainGroup = new THREE.Group();
scene.add(brainGroup);
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
raycaster.params.Line.threshold = 3;

function resizeBrain() {
  const rect = brainHost.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height, false);
  camera.aspect = rect.width / Math.max(rect.height, 1);
  camera.updateProjectionMatrix();
}

function roleStyle(role) {
  if (role === "DNa02") return { color: 0xfbbf24, opacity: .88 };
  if (role === "PFL3") return { color: 0x56d6ff, opacity: .44 };
  return { color: 0x94a3b8, opacity: .25 };
}

function materialFor(neuron) {
  const role = neuron.display_metadata?.role || "unclassified";
  const style = roleStyle(role);
  const material = new THREE.LineBasicMaterial({
    color: style.color,
    transparent: true,
    opacity: style.opacity,
    depthWrite: false,
  });
  material.userData = { baseOpacity: style.opacity, role };
  return material;
}

function applyAnatomyFilter(filter) {
  activeFilter = filter;
  for (const child of brainGroup.children) {
    const role = child.userData.role || "unclassified";
    const selected = filter === "ALL" || role === filter;
    child.material.opacity = selected ? child.material.userData.baseOpacity : .035;
  }
  filterButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.filter === filter);
  });
}

function addSkeletons(payload) {
  anatomy = payload;
  brainGroup.clear();
  const center = new THREE.Vector3();
  if (payload.bounds_um?.min && payload.bounds_um?.max) {
    center.fromArray(payload.bounds_um.min).add(new THREE.Vector3().fromArray(payload.bounds_um.max)).multiplyScalar(.5);
  }

  payload.neurons.forEach((neuron) => {
    const values = neuron.line_vertices_um.flatMap((point) => [
      point[0] - center.x,
      point[1] - center.y,
      point[2] - center.z,
    ]);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(values, 3));
    const lines = new THREE.LineSegments(geometry, materialFor(neuron));
    const meta = neuron.display_metadata || {};
    lines.userData = {
      bodyId: neuron.body_id,
      role: meta.role || "unclassified",
      readoutSide: meta.readout_side || null,
      column: meta.column ?? null,
      pbLabel: meta.pb_label || null,
    };
    brainGroup.add(lines);
  });

  const box = new THREE.Box3().setFromObject(brainGroup);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  if (Number.isFinite(sphere.radius) && sphere.radius > 0) {
    camera.position.set(0, 0, sphere.radius * 2.15);
    controls.target.copy(sphere.center);
    controls.minDistance = sphere.radius * .28;
    controls.maxDistance = sphere.radius * 7;
    raycaster.params.Line.threshold = Math.max(2, sphere.radius * .012);
  }
  const roleText = payload.role_counts
    ? ` · ${payload.role_counts.PFL3 || 0} PFL3 · ${payload.role_counts.DNa02 || 0} DNa02`
    : "";
  brainStatus.textContent = `${payload.neuron_count} exact body IDs${roleText}`;
  applyAnatomyFilter(activeFilter);
}

function drawPhaseCurve(mechanism) {
  if (!mechanism?.threshold_reports || !phaseCurve) return;
  const threshold = String(mechanism.primary_structural_threshold ?? 5);
  const rows = mechanism.threshold_reports[threshold]?.phase_reports;
  if (!rows) return;

  const entries = Object.entries(rows)
    .map(([phase, row]) => [Number(phase), Number(row.turn)])
    .filter(([phase, turn]) => Number.isFinite(phase) && Number.isFinite(turn))
    .sort((a, b) => a[0] - b[0]);
  if (entries.length < 2) return;

  const ctx = phaseCurve.getContext("2d");
  const { width, height, ratio } = fitCanvas(phaseCurve);
  ctx.clearRect(0, 0, width, height);
  const padX = 10 * ratio;
  const padY = 12 * ratio;
  const maxAbs = Math.max(.05, ...entries.map(([, turn]) => Math.abs(turn)));
  const minPhase = entries[0][0];
  const maxPhase = entries[entries.length - 1][0];
  const xFor = (phase) => padX + ((phase - minPhase) / Math.max(maxPhase - minPhase, 1)) * (width - 2 * padX);
  const yFor = (turn) => height / 2 - (turn / maxAbs) * (height / 2 - padY);

  ctx.strokeStyle = "rgba(148,163,184,.16)";
  ctx.lineWidth = ratio;
  ctx.beginPath();
  ctx.moveTo(padX, height / 2);
  ctx.lineTo(width - padX, height / 2);
  ctx.stroke();

  const fill = ctx.createLinearGradient(0, 0, width, 0);
  fill.addColorStop(0, "rgba(86,214,255,.12)");
  fill.addColorStop(.5, "rgba(251,191,36,.08)");
  fill.addColorStop(1, "rgba(251,113,133,.10)");
  ctx.beginPath();
  entries.forEach(([phase, turn], index) => {
    const x = xFor(phase);
    const y = yFor(turn);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(xFor(maxPhase), height / 2);
  ctx.lineTo(xFor(minPhase), height / 2);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();

  ctx.strokeStyle = "rgba(251,191,36,.92)";
  ctx.lineWidth = 1.7 * ratio;
  ctx.beginPath();
  entries.forEach(([phase, turn], index) => {
    const x = xFor(phase);
    const y = yFor(turn);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  for (const [phase, turn] of entries) {
    if (phase !== 0) continue;
    ctx.fillStyle = "#fef3c7";
    ctx.beginPath();
    ctx.arc(xFor(phase), yFor(turn), 2.4 * ratio, 0, Math.PI * 2);
    ctx.fill();
  }
}

function updateMechanismSummary(mechanism) {
  if (!mechanism) {
    mechanismTitle.textContent = "No mechanism report loaded";
    mechanismPass.textContent = "ANATOMY ONLY";
    mechanismCopy.textContent = "The renderer will not infer modeled neural state from morphology alone.";
    qualificationPill.textContent = "no probe";
    return;
  }
  const threshold = mechanism.primary_structural_threshold ?? "?";
  const gateText = mechanism.gate_count != null
    ? `${mechanism.passed_gate_count ?? "?"}/${mechanism.gate_count} gates`
    : mechanism.passed ? "passed" : "not passed";
  mechanismTitle.textContent = `${mechanism.protocol || "mechanism"} · t=${threshold}`;
  mechanismPass.textContent = mechanism.passed ? "QUALIFIED PROBE" : "NOT QUALIFIED";
  mechanismPass.classList.toggle("qualified", mechanism.passed);
  mechanismCopy.textContent = `Modeled phase → descending turn · ${gateText}. This computation is not synchronized to the behavior replay.`;
  qualificationPill.textContent = mechanism.passed ? `${gateText} · t=${threshold}` : "probe not qualified";
  drawPhaseCurve(mechanism);
}

async function loadOptionalSkeletons() {
  try {
    const response = await fetch(ASSET_URL);
    if (!response.ok) throw new Error(`${response.status}`);
    const payload = await response.json();
    addSkeletons(payload);
  } catch (error) {
    brainStatus.textContent = "exact morphology not loaded";
    evidenceNote.textContent = "Connectome asset missing. Run the SWC packer, then copy the JSON into web/connectome-twin/public/data/.";
  }
}

async function loadStream() {
  const response = await fetch(STREAM_URL);
  if (!response.ok) throw new Error(`replay stream unavailable: ${response.status}`);
  stream = await response.json();
  const behavior = stream.timelines.behavior;
  const mechanism = stream.timelines.mechanism;
  if (mechanism?.synchronized_with_behavior === false) {
    timelineWarning.textContent = "mechanism probe · separate clock";
  }
  updateMechanismSummary(mechanism);
  evidenceNote.textContent = behavior.claim_boundary;
  behaviorIndex = 0;
  durationLabel.textContent = behavior.frames.length
    ? `${behavior.frames[behavior.frames.length - 1].t_s.toFixed(1)} s`
    : "replay";
  drawWorld(behavior.frames[0], behavior.world, 0);
}

function updateNeuronHover(event) {
  if (!anatomy) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(brainGroup.children, false);
  const hit = hits.find((entry) => entry.object.material.opacity > .08);
  if (!hit) {
    neuronHover.textContent = activeFilter === "ALL" ? "hover a neuron" : `filter · ${activeFilter}`;
    return;
  }
  const data = hit.object.userData;
  const detail = [data.role, `body ${data.bodyId}`];
  if (data.column != null) detail.push(`C${data.column}`);
  if (data.pbLabel) detail.push(data.pbLabel);
  neuronHover.textContent = detail.filter(Boolean).join(" · ");
}

function tick(now) {
  requestAnimationFrame(tick);
  controls.update();
  renderer.render(scene, camera);
  if (!stream || !playing) return;
  const behavior = stream.timelines.behavior;
  if (now - lastTick > 33) {
    behaviorIndex = (behaviorIndex + 1) % behavior.frames.length;
    scrubber.value = Math.round((behaviorIndex / Math.max(behavior.frames.length - 1, 1)) * 1000);
    drawWorld(behavior.frames[behaviorIndex], behavior.world, behaviorIndex);
    lastTick = now;
  }
}

filterButtons.forEach((button) => {
  button.addEventListener("click", () => applyAnatomyFilter(button.dataset.filter || "ALL"));
});

renderer.domElement.addEventListener("pointermove", updateNeuronHover);
renderer.domElement.addEventListener("pointerleave", () => {
  neuronHover.textContent = activeFilter === "ALL" ? "hover a neuron" : `filter · ${activeFilter}`;
});

scrubber.addEventListener("input", () => {
  if (!stream) return;
  const frames = stream.timelines.behavior.frames;
  behaviorIndex = Math.round((Number(scrubber.value) / 1000) * (frames.length - 1));
  drawWorld(frames[behaviorIndex], stream.timelines.behavior.world, behaviorIndex);
});

playButton.addEventListener("click", () => {
  playing = !playing;
  playIcon.textContent = playing ? "Ⅱ" : "▶";
  playButton.setAttribute("aria-label", playing ? "Pause replay" : "Play replay");
});

window.addEventListener("resize", () => {
  resizeBrain();
  drawPhaseCurve(stream?.timelines?.mechanism);
  if (stream) drawWorld(stream.timelines.behavior.frames[behaviorIndex], stream.timelines.behavior.world, behaviorIndex);
});

resizeBrain();
Promise.all([loadStream(), loadOptionalSkeletons()]).catch((error) => {
  evidenceNote.textContent = error.message;
});
requestAnimationFrame(tick);
