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
const turnCommand = document.querySelector("#turn-command");
const actionLabel = document.querySelector("#action-label");
const timelineWarning = document.querySelector("#timeline-warning");
const scrubber = document.querySelector("#scrubber");
const playButton = document.querySelector("#play-button");

let stream = null;
let behaviorIndex = 0;
let playing = true;
let lastTick = performance.now();

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

function worldToCanvas(x, y, world, width, height, pad = 46) {
  const sx = (width - 2 * pad) / world.width;
  const sy = (height - 2 * pad) / world.height;
  return [pad + x * sx, height - pad - y * sy];
}

function drawWorld(frame, world) {
  const ctx = worldCanvas.getContext("2d");
  const { width, height, ratio } = fitCanvas(worldCanvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#090f1c";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(148,163,184,.10)";
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

  const complete = frame.plume_snapshot_complete;
  for (const [x, y, sigma] of frame.plume_components) {
    const [px, py] = worldToCanvas(x, y, world, width, height);
    const radius = Math.max(1.5 * ratio, sigma * 8 * ratio);
    const grad = ctx.createRadialGradient(px, py, 0, px, py, radius * 2.8);
    grad.addColorStop(0, complete ? "rgba(163,230,53,.20)" : "rgba(163,230,53,.10)");
    grad.addColorStop(0.4, "rgba(101,163,13,.09)");
    grad.addColorStop(1, "rgba(101,163,13,0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(px, py, radius * 2.8, 0, Math.PI * 2);
    ctx.fill();
  }

  if (world.source_for_viewer_only) {
    const [sx, sy] = worldToCanvas(
      world.source_for_viewer_only[0],
      world.source_for_viewer_only[1],
      world,
      width,
      height,
    );
    ctx.strokeStyle = "rgba(217,249,157,.45)";
    ctx.lineWidth = 1.5 * ratio;
    ctx.beginPath();
    ctx.arc(sx, sy, 8 * ratio, 0, Math.PI * 2);
    ctx.stroke();
  }

  frame.agents.forEach((agent, idx) => {
    const [x, y] = worldToCanvas(agent.position[0], agent.position[1], world, width, height);
    const color = idx === 0 ? "#38bdf8" : "#fb7185";
    const heading = -agent.heading_rad;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(heading);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = idx === 0 ? 18 * ratio : 8 * ratio;
    ctx.beginPath();
    ctx.moveTo(13 * ratio, 0);
    ctx.lineTo(-8 * ratio, -7 * ratio);
    ctx.lineTo(-4 * ratio, 0);
    ctx.lineTo(-8 * ratio, 7 * ratio);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  });

  const primary = frame.agents[0];
  behaviorClock.textContent = `${frame.t_s.toFixed(2)} s`;
  odorLeft.textContent = primary.sensors.odor_left.toFixed(2);
  odorRight.textContent = primary.sensors.odor_right.toFixed(2);
  turnCommand.textContent = primary.command.turn.toFixed(2);
  const turn = primary.command.turn;
  actionLabel.textContent = Math.abs(turn) < 0.08 ? "STRAIGHT" : turn > 0 ? "TURN LEFT" : "TURN RIGHT";
}

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x070b12, 0.0018);
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100000);
camera.position.set(0, 0, 1200);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x000000, 0);
brainHost.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.35;
controls.minDistance = 100;
controls.maxDistance = 5000;
const brainGroup = new THREE.Group();
scene.add(brainGroup);

function resizeBrain() {
  const rect = brainHost.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height, false);
  camera.aspect = rect.width / Math.max(rect.height, 1);
  camera.updateProjectionMatrix();
}

function materialFor(index) {
  const palette = [0x94a3b8, 0x67e8f9, 0xc4b5fd, 0xfde68a, 0xa3e635, 0xfb7185];
  return new THREE.LineBasicMaterial({
    color: palette[index % palette.length],
    transparent: true,
    opacity: index === 0 ? 0.78 : 0.58,
  });
}

function addSkeletons(payload) {
  brainGroup.clear();
  const center = new THREE.Vector3();
  if (payload.bounds_um?.min && payload.bounds_um?.max) {
    center.fromArray(payload.bounds_um.min).add(new THREE.Vector3().fromArray(payload.bounds_um.max)).multiplyScalar(0.5);
  }
  payload.neurons.forEach((neuron, index) => {
    const values = neuron.line_vertices_um.flatMap((point) => [point[0] - center.x, point[1] - center.y, point[2] - center.z]);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(values, 3));
    const lines = new THREE.LineSegments(geometry, materialFor(index));
    lines.userData.bodyId = neuron.body_id;
    brainGroup.add(lines);
  });
  const box = new THREE.Box3().setFromObject(brainGroup);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  if (Number.isFinite(sphere.radius) && sphere.radius > 0) {
    camera.position.set(0, 0, sphere.radius * 2.35);
    controls.target.copy(sphere.center);
    controls.minDistance = sphere.radius * 0.35;
    controls.maxDistance = sphere.radius * 8;
  }
  brainStatus.textContent = `${payload.neuron_count} exact body IDs`;
}

async function loadOptionalSkeletons() {
  try {
    const response = await fetch(ASSET_URL);
    if (!response.ok) throw new Error(`${response.status}`);
    const payload = await response.json();
    addSkeletons(payload);
  } catch (error) {
    brainStatus.textContent = "add route-skeletons-v1.json";
    evidenceNote.textContent = "Connectome asset missing. Run the SWC packer, then copy the JSON into web/connectome-twin/public/data/.";
  }
}

async function loadStream() {
  const response = await fetch(STREAM_URL);
  if (!response.ok) throw new Error(`replay stream unavailable: ${response.status}`);
  stream = await response.json();
  const behavior = stream.timelines.behavior;
  if (stream.timelines.mechanism?.synchronized_with_behavior === false) {
    timelineWarning.textContent = "mechanism probe uses a separate clock";
  }
  evidenceNote.textContent = behavior.claim_boundary;
  behaviorIndex = 0;
  drawWorld(behavior.frames[0], behavior.world);
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
    drawWorld(behavior.frames[behaviorIndex], behavior.world);
    lastTick = now;
  }
}

scrubber.addEventListener("input", () => {
  if (!stream) return;
  const frames = stream.timelines.behavior.frames;
  behaviorIndex = Math.round((Number(scrubber.value) / 1000) * (frames.length - 1));
  drawWorld(frames[behaviorIndex], stream.timelines.behavior.world);
});

playButton.addEventListener("click", () => {
  playing = !playing;
  playButton.textContent = playing ? "Pause" : "Play";
});

window.addEventListener("resize", () => {
  resizeBrain();
  if (stream) drawWorld(stream.timelines.behavior.frames[behaviorIndex], stream.timelines.behavior.world);
});

resizeBrain();
Promise.all([loadStream(), loadOptionalSkeletons()]).catch((error) => {
  evidenceNote.textContent = error.message;
});
requestAnimationFrame(tick);
