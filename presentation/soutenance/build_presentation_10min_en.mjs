import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const C = {
  navy: "#0B1F33",
  navy2: "#153B5B",
  teal: "#1FA6A0",
  tealDark: "#137B77",
  tealPale: "#DDF3F1",
  orange: "#F28C45",
  orangePale: "#FCE8D9",
  red: "#C94F4F",
  redPale: "#F8DEDE",
  cream: "#F7F8FA",
  white: "#FFFFFF",
  ink: "#142632",
  muted: "#5A6C79",
  light: "#DCE4E9",
  pale: "#EDF2F5",
  green: "#3B8F68",
};

const FONT = "Aptos";
const SLIDE_W = 1280;
const SLIDE_H = 720;

function argsFrom(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    if (!argv[i].startsWith("--")) continue;
    out[argv[i].slice(2)] = argv[i + 1];
    i += 1;
  }
  return out;
}

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function readBytes(filePath) {
  const bytes = await fs.readFile(filePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function box(slide, x, y, w, h, fill = C.white, line = C.light, radius = 18, name) {
  return slide.shapes.add({
    geometry: "roundRect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: line === "none" ? 0 : 1 },
    borderRadius: radius,
  });
}

function rect(slide, x, y, w, h, fill, name) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function text(slide, value, x, y, w, h, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: options.name,
    position: { left: x, top: y, width: w, height: h, rotation: options.rotation || 0 },
    fill: options.fill || "none",
    line: { style: "solid", fill: options.line || "none", width: options.line ? 1 : 0 },
    borderRadius: options.radius || 0,
  });
  shape.text = value;
  shape.text.style = {
    typeface: options.typeface || FONT,
    fontSize: options.size || 24,
    bold: Boolean(options.bold),
    italic: Boolean(options.italic),
    color: options.color || C.ink,
    alignment: options.align || "left",
    verticalAlignment: options.valign || "middle",
    autoFit: options.autoFit || "shrinkText",
    wrap: "square",
    lineSpacing: options.lineSpacing || 1.0,
    insets: options.insets || { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return shape;
}

function circle(slide, cx, cy, d, fill, label = "", options = {}) {
  const shape = slide.shapes.add({
    geometry: "ellipse",
    name: options.name,
    position: { left: cx - d / 2, top: cy - d / 2, width: d, height: d },
    fill,
    line: { style: "solid", fill: options.line || C.white, width: options.lineWidth || 3 },
    shadow: options.shadow || "shadow-sm",
  });
  if (label) {
    shape.text = label;
    shape.text.style = {
      typeface: FONT,
      fontSize: options.size || 22,
      bold: true,
      color: options.color || C.white,
      alignment: "center",
      verticalAlignment: "middle",
      autoFit: "shrinkText",
      insets: { top: 0, right: 0, bottom: 0, left: 0 },
    };
  }
  return shape;
}

function connector(slide, from, to, options = {}) {
  return slide.shapes.connect(from, to, {
    kind: options.kind || "straight",
    fromSide: options.fromSide,
    toSide: options.toSide,
    line: {
      style: options.dashed ? "dashed" : "solid",
      fill: options.color || C.teal,
      width: options.width || 4,
    },
    head: { type: "none" },
    tail: options.arrow === false ? { type: "none" } : { type: "triangle", width: "sm", length: "sm" },
  });
}

function line(slide, x1, y1, x2, y2, color = C.light, width = 2, dashed = false) {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  const lineWidth = Math.abs(x2 - x1);
  const lineHeight = Math.abs(y2 - y1);
  return slide.shapes.add({
    geometry: "line",
    position: {
      left,
      top,
      width: lineWidth,
      height: lineHeight,
      verticalFlip: (x2 >= x1 && y2 < y1) || (x2 < x1 && y2 >= y1),
    },
    fill: "none",
    line: { style: dashed ? "dashed" : "solid", fill: color, width },
  });
}

function header(slide, title, number, backup = false, section = "SMART VISIT / PFE DEFENSE") {
  slide.background.fill = C.cream;
  text(slide, backup ? "BACKUP" : section, 64, 28, 650, 22, {
    size: 13,
    bold: true,
    color: backup ? C.orange : C.tealDark,
    name: "eyebrow",
  });
  text(slide, title, 64, 62, 1152, 58, {
    size: title.length > 60 ? 35 : 38,
    bold: true,
    color: C.navy,
    name: "slide-title",
  });
  rect(slide, 64, 128, 92, 5, backup ? C.orange : C.teal, "title-accent");
  line(slide, 64, 676, 1216, 676, C.light, 1);
  text(slide, "Ayman Naaimi · ENSAM Meknes", 64, 688, 440, 18, { size: 11, color: C.muted });
  text(slide, backup ? `Backup ${number - 10}` : `${number} / 10`, 1110, 688, 106, 18, {
    size: 11,
    color: C.muted,
    align: "right",
  });
}

function source(slide, value) {
  text(slide, value, 64, 652, 1120, 17, { size: 10, color: C.muted, italic: true });
}

function noteText(item) {
  return [
    `TIME: ${item.time}`,
    `MEMORY SENTENCE: ${item.memory}`,
    "",
    "SCRIPT:",
    item.script,
    "",
    `TRANSITION: ${item.transition}`,
    "",
    `SKIP IF LATE: ${item.skip}`,
  ].join("\n");
}

const TALK_LEGACY = [
  {
    title: "Smart Visit: From 5,100 Places to One Coherent Itinerary",
    time: "0:25",
    script: "Good morning. Smart Visit turns about 5,100 candidate places into one personalized, ordered itinerary. I will show the prediction engine, the simple route decoder, and the two separate evaluations that keep the conclusions fair.",
    memory: "Smart Visit turns 5,100 places into one personalized route.",
    transition: "First, what does the traveler need?",
    skip: "It is a reproducible contribution, not a record claim.",
  },
  {
    title: "A Good Trip Answers Two Questions",
    time: "0:45",
    script: "A traveler needs more than attractive places. They need to know what to visit and in what order. First, which place is most likely to come next? Second, can repeated choices form a useful route from start to finish? A model can predict the next place well and still create revisits or a poor order. This thesis addresses that gap.",
    memory: "Next-POI asks what comes next; itinerary generation asks what order works.",
    transition: "This leads to a two-phase system and a two-benchmark evaluation.",
    skip: "In other words, local accuracy does not automatically produce global route quality.",
  },
  {
    title: "One Engine, One Decoder, and Two Fair Evaluations",
    time: "0:55",
    script: "The study has two phases. Phase 1 learns on Foursquare New York and ranks about 5,100 places, using HR at k, NDCG, and MRR. Phase 2 uses Frozen Rollout to build complete itineraries. Routes are compared internally on New York, then evaluated on the Flickr Benchmark under Chen 2016. Here the metrics are set-F1 and order-aware pairs-F1. The two tasks measure different outcomes, so their scores must never share one leaderboard.",
    memory: "One engine and one decoder are evaluated on two strictly separate benchmarks.",
    transition: "With that map in mind, let us open the prediction engine.",
    skip: "The rule is simple: compare only the same dataset, protocol, task, and metric.",
  },
  {
    title: "The Engine Combines Places, Context, and User Preferences",
    time: "1:15",
    script: "The engine combines four components. A hybrid graph links places that are often visited consecutively or geographically close; a two-layer GCN turns it into POI features. The graph uses training-only co-visitation links and each place's ten nearest geographic neighbors. Distance and elapsed time add context. A GRU summarizes the visited sequence. A learned user embedding adds identity-based preference information; it is learned from visit history, not from a written persona. The GRU state and user embedding pass through an MLP that scores all 5,100 places. Every session contributes every prefix-to-next-place pair, giving roughly one hundred thousand training examples.",
    memory: "Graph, context, sequence, and user identity produce the next-POI ranking.",
    transition: "Now we can ask how accurate this compact engine is.",
    skip: "The implemented context is limited to distance and elapsed time; weather and time of day remain future work.",
  },
  {
    title: "Competitive, but Not State of the Art",
    time: "0:50",
    script: "On New York, HR at one is 0.187: the exact next place ranks first almost once in five among about 5,100 candidates. HR at ten is 0.588, so it appears in the top ten almost six times in ten. LSTM scores 0.130 and STGCN 0.180; our 0.187 is higher, while attention, transformer, and LLM methods remain ahead.",
    memory: "HR at one is 0.187: an honest LSTM-STGCN-tier result.",
    transition: "The next question is how to turn these local scores into a route.",
    skip: "This is competitive in the LSTM-STGCN tier, but not state of the art.",
  },
  {
    title: "Frozen Rollout Builds a Loop-Free Route",
    time: "1:15",
    script: "Frozen Rollout receives the user, start, destination, and desired number of stops. At each step, the frozen engine scores every place. The decoder masks visited places, reserves the destination for the final position, selects the best remaining candidate, and repeats. This guarantees a loop-free route with the requested endpoints and length. Greedy decoding takes the best candidate; beam search keeps several partial routes. Both enforce the same constraints. No model weight changes and no additional training is needed: the same checkpoint now serves the itinerary task. The itinerary capability is added entirely at inference time.",
    memory: "Frozen Rollout builds loop-free routes without retraining the engine.",
    transition: "I then tested whether a model trained directly for itineraries could do better.",
    skip: "The headline method does not claim a global optimum or travel-time optimization.",
  },
  {
    title: "Dense Supervision Beats a Purpose-Trained Model",
    time: "1:05",
    script: "On 2,880 New York test routes of length at least three, Frozen Rollout reaches 0.289 pairs-F1. The Trained Pointer, trained end to end on whole routes, reaches 0.259, or 0.261 with context. Validation shows this is a genuine negative result, not a bug. The main explanation is supervision density: the engine sees about one hundred thousand prefix-target examples, while the Trained Pointer sees one whole trajectory per session. Beam search moves Frozen Rollout only to 0.290, so extra search cannot replace missing information in the scorer.",
    memory: "Dense prefix supervision lets Frozen Rollout beat the Trained Pointer.",
    transition: "However, these New York values cannot be compared directly with published itinerary scores.",
    skip: "Set-F1 of 0.609 suggests that finding places is easier than ordering them.",
  },
  {
    title: "A Fair Comparison Requires the Same Benchmark",
    time: "0:55",
    script: "Flickr city vocabularies are far smaller than New York, so the chance level is different. For a fair bridge, I reproduced Chen 2016: leave one trajectory out, give the first and last POI plus length, and evaluate loop-free routes of length at least three. Random is the key check because it has no modeling choices. Across five cities, my Random score differs from Chen by at most 0.021. This acts like a unit test for the complete evaluation protocol.",
    memory: "Reproducing Random within 0.021 validates the Flickr evaluation protocol.",
    transition: "Only after this check is a literature comparison meaningful.",
    skip: "That match indicates that the data, split, query, and pairs-F1 calculation are aligned.",
  },
  {
    title: "The Benchmark Reveals Both Positioning and Cold-Start",
    time: "1:10",
    script: "On the Flickr Benchmark, our methods span 0.23 to 0.59 pairs-F1 on the published scale. Markov is strongest among ours, reaching 0.587 in Glasgow. The Trained Pointer beats Random but trails Markov in every city, while SelfTrip and AR-Trip remain around 0.8. Our classical methods therefore reach the literature's scale, but the richer neural methods remain clearly ahead. Personalization is mixed: the user embedding changes pairs-F1 by plus 0.038 in Glasgow, minus 0.007 in Osaka, and minus 0.017 in Toronto. Under leave-one-out, many users offer too little history for a stable identity embedding, which is a cold-start effect.",
    memory: "Our methods reach the published scale, but cold-start limits identity-based personalization.",
    transition: "I will finish with the three results worth remembering.",
    skip: "A content- or persona-based signal is therefore a logical next step.",
  },
  {
    title: "Three Results to Remember",
    time: "0:40",
    script: "Three results matter. The engine reaches HR at one of 0.187 on 5,100 places. Frozen Rollout builds loop-free routes and beats the Trained Pointer, 0.289 versus 0.259. The Flickr Benchmark closely reproduces Chen's Random baseline, so the literature positioning is defensible. A trained local model can support itinerary generation when decoding constraints and evaluation boundaries are explicit. Thank you.",
    memory: "Remember 0.187, 0.289 versus 0.259, and the validated Flickr bridge.",
    transition: "That concludes the presentation and opens the discussion.",
    skip: "I am ready for your questions.",
  },
];

const TALK = [
  {
    title: "Smart Visit: From 5,100 Places to One Coherent Itinerary",
    time: "0:25",
    script: "Good morning. Smart Visit asks how roughly 5,100 candidate places can become one personalized, coherent tourist itinerary. I will follow five parts: the problem, the state of the art, the solution, its implementation, and the conclusion.",
    memory: "Smart Visit converts candidates into one coherent route.",
    transition: "First, what problem are we solving?",
    skip: "I will follow five parts in this presentation.",
  },
  {
    title: "A Good Trip Answers Two Questions",
    time: "0:55",
    script: "A tourist system must answer two different questions: what should this user visit next, and in what order should several places form a complete trip? A ranked list is not yet an itinerary; it may repeat locations, ignore the destination, or create a poor order. The research question is whether a personalized, context-aware next-POI engine can be decoded into a loop-free route without confusing next-step accuracy with itinerary quality.",
    memory: "A useful trip needs both selection and correct order.",
    transition: "Now consider how the literature approaches this problem.",
    skip: "A ranked list is not yet an itinerary.",
  },
  {
    title: "Existing Methods Trade Off Personalization and Feasibility",
    time: "1:00",
    script: "Three broad families dominate the literature. Route-optimization methods such as OP and TTDP enforce constraints by construction, but often use simple utility scores. Personalized and context-aware recommenders learn relevance, yet usually rank individual POIs rather than generate a constrained route. Deep sequential models learn mobility transitions, but next-POI objectives and greedy decoding do not guarantee global coherence. No family fully unifies personalization, context, ordering, and route feasibility. The central gaps are therefore the connection between prediction and itinerary quality, and fragmented evaluation. This thesis addresses those two gaps directly; cold-start and richer environmental context remain partial gaps.",
    memory: "Existing methods trade off personalization and route feasibility.",
    transition: "These gaps determine the proposed solution.",
    skip: "Cold-start and richer environmental context remain partial gaps.",
  },
  {
    title: "One Engine, One Decoder, and Two Fair Evaluations",
    time: "1:00",
    script: "The solution deliberately separates two phases. Phase one trains a personalized next-POI engine on Foursquare New York and evaluates it with HR, NDCG, and MRR. Phase two turns its scores into routes through Frozen Rollout, compared internally with a Trained Pointer using set-F1 and pairs-F1. Since New York itinerary values cannot be compared with Flickr literature values, the itinerary task is repeated on the Flickr Benchmark under the Chen-2016 protocol. The two tasks never share one leaderboard.",
    memory: "One pipeline contains two distinct evaluation spaces.",
    transition: "Let us now open the prediction engine.",
    skip: "The two tasks never share one leaderboard.",
  },
  {
    title: "The Engine Combines Places, Context, and User Preferences",
    time: "1:10",
    script: "The implemented engine combines four signals. A hybrid graph links POIs through training-only co-visits and ten geographic neighbours, and a two-layer GCN creates place features. Distance and elapsed-time gaps provide lightweight context. A GRU summarizes the visited sequence, while a learned user embedding represents identity-based preference. The GRU state and user embedding feed an MLP that scores the full vocabulary of about 5,100 POIs. Each session contributes every prefix-to-next-POI pair, producing on the order of one hundred thousand supervised examples. Training uses Adam, dropout, gradient clipping, and early stopping on validation HR at ten.",
    memory: "Graph, context, user, and sequence signals produce one ranking.",
    transition: "The same trained engine is then reused as a route scorer.",
    skip: "Training uses Adam, dropout, clipping, and early stopping.",
  },
  {
    title: "Frozen Rollout Turns Scores into a Loop-Free Route",
    time: "1:05",
    script: "Frozen Rollout creates an itinerary without retraining the engine. A query provides a user, a start POI, an optional destination, and the desired length. At each step, the frozen engine scores all candidates; the decoder masks visited POIs, reserves a supplied destination for the final step, selects the best valid candidate, and repeats. Greedy is the headline decoder, while beam search keeps several partial routes. When a destination is supplied, the method guarantees it as the last stop; in all cases, the route is loop-free and has the requested length. It does not claim a global optimum, opening-hour feasibility, or travel-time optimization.",
    memory: "Frozen Rollout adds route constraints without changing the model.",
    transition: "We can now evaluate the engine and the generated routes.",
    skip: "Beam search keeps several partial routes.",
  },
  {
    title: "Competitive, but Not State of the Art",
    time: "0:45",
    script: "On New York, the engine obtains HR at one of 0.187, HR at five of 0.476, HR at ten of 0.588, and MRR of 0.316. HR at one means the exact next POI ranks first among roughly 5,100 candidates almost one time in five. The result is slightly above STGCN at 0.180, but below GETNext at 0.240 and LLM4POI at 0.340. Its honest position is the LSTM/STGCN tier, not the state of the art.",
    memory: "The engine is competitive, but not state of the art.",
    transition: "The itinerary experiment gives the more surprising result.",
    skip: "Its honest position is the LSTM-STGCN tier.",
  },
  {
    title: "Frozen Rollout Wins; Dense Supervision Helps Explain Why",
    time: "1:00",
    script: "On 2,880 New York itineraries of length at least three, Frozen Rollout reaches 0.289 pairs-F1, compared with 0.259 for the Trained Pointer. Beam search reaches only 0.290. The strongest explanation is supervision density: the engine learns from every prefix, on the order of one hundred thousand prefix-target examples, whereas the Trained Pointer sees one complete trajectory per session and trains from scratch. The lesson is not that pointer models never work; it is that a densely supervised scorer is difficult to beat. These itinerary metrics are never compared with next-POI HR.",
    memory: "Frozen Rollout wins, and dense prefix supervision helps explain why.",
    transition: "But New York is not a valid literature benchmark.",
    skip: "Beam search reaches only 0.290.",
  },
  {
    title: "Flickr Validates the Protocol and Shows the Remaining Gap",
    time: "1:15",
    script: "For fair literature comparison, the Flickr Benchmark follows the Chen-2016 protocol: leave one trajectory out, provide the first and last POI plus route length, and evaluate loop-free routes of length at least three with pairs-F1. Our Random results reproduce Chen's published Random baseline within a maximum absolute gap of 0.021, validating the harness. On that published scale, our classical baselines span 0.23 to 0.59, and the Trained Pointer spans 0.31 to 0.49, while SelfTrip and AR-Trip remain around 0.80. Personalization is conditional: the user embedding changes pairs-F1 by plus 0.038 in Glasgow, minus 0.007 in Osaka, and minus 0.017 in Toronto. These deltas come from a separate user-embedding ablation, not the main pointer run. Sparse leave-one-out history therefore exposes cold-start. Other limits are New-York-only engine training, offline evaluation, context restricted to distance and elapsed time, and no opening-hour or scheduling constraints. New York and Flickr scores are never compared directly.",
    memory: "Flickr makes the literature comparison valid and honest.",
    transition: "I will close with the three results to retain.",
    skip: "New York and Flickr scores are never compared directly.",
  },
  {
    title: "Three Results to Remember",
    time: "0:40",
    script: "Three results matter. The engine reaches HR at one of 0.187. Frozen Rollout converts its scores into loop-free itineraries and reaches 0.289 pairs-F1, above the Trained Pointer at 0.259 on the same New York protocol. The Flickr reproduction validates the literature comparison. The contribution is therefore a reproducible foundation: a personalized scorer, a simple effective decoder, and an honest evaluation. Thank you.",
    memory: "Score, decode, and evaluate honestly.",
    transition: "Thank you; I welcome your questions.",
    skip: "The contribution is a reproducible foundation for future work.",
  },
];

function setTalkNotes(slide, index) {
  slide.speakerNotes.textFrame.setText(noteText(TALK[index]));
  slide.speakerNotes.setVisible(true);
}

function buildSlide1(presentation, heroBytes) {
  const slide = presentation.slides.add();
  slide.background.fill = C.navy;
  slide.images.add({
    blob: heroBytes,
    contentType: "image/png",
    alt: "Abstract dark city map with a highlighted five-stop tourist route",
    fit: "cover",
    position: { left: 0, top: 0, width: SLIDE_W, height: SLIDE_H },
  });
  rect(slide, 68, 62, 94, 6, C.orange);
  text(slide, "SMART VISIT", 68, 84, 300, 30, { size: 17, bold: true, color: C.tealPale });
  text(slide, "From 5,100 places\nto one coherent itinerary", 68, 150, 610, 185, {
    size: 55,
    bold: true,
    color: C.white,
    lineSpacing: 0.92,
    valign: "top",
    name: "cover-title",
  });
  text(slide, "Personalized tourist itinerary recommendation\nbased on user preferences and contextual data", 72, 365, 535, 78, {
    size: 23,
    color: "#D6E4EE",
    valign: "top",
  });
  text(slide, "Ayman Naaimi", 72, 548, 330, 34, { size: 23, bold: true, color: C.white });
  text(slide, "Supervisor: Imadeddine Mountasser\nMaster D2SM · ENSAM Meknes · 2025-2026", 72, 590, 500, 55, {
    size: 16,
    color: "#B8C8D5",
    valign: "top",
  });
  setTalkNotes(slide, 0);
  return slide;
}

function buildSlide2(presentation) {
  const slide = presentation.slides.add();
  header(slide, "A good trip answers two questions: what to visit, and in what order?", 2, false, "1 · PROBLEM");

  text(slide, "1", 96, 170, 36, 36, { size: 24, bold: true, color: C.white, align: "center", fill: C.navy, radius: 18 });
  text(slide, "WHAT TO VISIT", 145, 170, 360, 36, { size: 19, bold: true, color: C.navy });
  box(slide, 76, 220, 430, 342, C.pale, C.light, 24);
  const dots = [
    [125, 266, C.navy2], [180, 250, C.teal], [240, 284, C.muted], [305, 252, C.navy2], [380, 274, C.teal], [454, 246, C.muted],
    [145, 330, C.teal], [215, 342, C.navy2], [285, 318, C.muted], [348, 352, C.teal], [430, 334, C.navy2],
    [116, 407, C.muted], [178, 432, C.navy2], [246, 398, C.teal], [318, 430, C.muted], [390, 406, C.teal], [457, 438, C.navy2],
    [144, 498, C.navy2], [226, 486, C.muted], [302, 512, C.teal], [374, 486, C.navy2], [446, 510, C.muted],
  ];
  for (const [x, y, color] of dots) circle(slide, x, y, 14, color, "", { line: color, lineWidth: 0, shadow: "shadow-none" });
  text(slide, "Rank thousands of candidates", 120, 524, 340, 25, { size: 18, color: C.muted, align: "center" });

  const arrow = slide.shapes.add({
    geometry: "rightArrow",
    position: { left: 530, top: 350, width: 120, height: 70 },
    fill: C.orange,
    line: { style: "solid", fill: C.orange, width: 0 },
  });
  arrow.text = "choose\n+ order";
  arrow.text.style = { typeface: FONT, fontSize: 17, bold: true, color: C.white, alignment: "center", verticalAlignment: "middle" };

  text(slide, "2", 700, 170, 36, 36, { size: 24, bold: true, color: C.white, align: "center", fill: C.tealDark, radius: 18 });
  text(slide, "IN WHAT ORDER", 749, 170, 360, 36, { size: 19, bold: true, color: C.navy });
  box(slide, 680, 220, 524, 342, C.tealPale, "#B8DEDA", 24);
  const route = [
    circle(slide, 745, 470, 45, C.tealDark, "S"),
    circle(slide, 820, 390, 45, C.teal, "1"),
    circle(slide, 915, 430, 45, C.teal, "2"),
    circle(slide, 1008, 320, 45, C.orange, "3"),
    circle(slide, 1128, 280, 45, C.tealDark, "E"),
  ];
  for (let i = 0; i < route.length - 1; i += 1) connector(slide, route[i], route[i + 1], { color: C.tealDark, width: 5 });
  text(slide, "One ordered, loop-free route", 748, 520, 390, 26, { size: 18, color: C.tealDark, align: "center", bold: true });
  source(slide, "Conceptual illustration based on the thesis problem definition.");
  setTalkNotes(slide, 1);
  return slide;
}

function buildSlide4(presentation) {
  const slide = presentation.slides.add();
  header(slide, "One engine, one decoder, and two fair evaluations", 4, false, "3 · PROPOSED SOLUTION");

  text(slide, "PHASE 1 · NEXT-POI", 78, 168, 290, 28, { size: 17, bold: true, color: C.navy });
  const d = box(slide, 78, 214, 225, 115, C.pale, C.light, 18);
  const e = box(slide, 390, 214, 255, 115, C.navy, C.navy, 18);
  const r = box(slide, 730, 214, 225, 115, C.tealPale, "#A8D7D2", 18);
  text(slide, "Foursquare NYC\n~5,100 POIs", 100, 236, 180, 66, { size: 22, bold: true, color: C.navy, align: "center" });
  text(slide, "Context-aware\nnext-POI engine", 420, 236, 195, 66, { size: 23, bold: true, color: C.white, align: "center" });
  text(slide, "Full-vocabulary\nPOI ranking", 756, 236, 175, 66, { size: 22, bold: true, color: C.tealDark, align: "center" });
  connector(slide, d, e, { fromSide: "right", toSide: "left", color: C.navy2 });
  connector(slide, e, r, { fromSide: "right", toSide: "left", color: C.teal });
  const evalNextPoi = box(slide, 988, 220, 218, 102, C.white, C.light, 18);
  text(slide, "Evaluate with\nHR@k · NDCG · MRR", 1006, 242, 182, 58, { size: 20, bold: true, color: C.navy, align: "center" });
  connector(slide, r, evalNextPoi, { fromSide: "right", toSide: "left", color: C.tealDark });

  text(slide, "PHASE 2 · ITINERARY", 78, 390, 290, 28, { size: 17, bold: true, color: C.orange });
  const q = box(slide, 78, 438, 225, 115, C.orangePale, "#F3C49D", 18);
  const f = box(slide, 390, 438, 255, 115, C.orange, C.orange, 18);
  const i = box(slide, 730, 438, 225, 115, C.tealPale, "#A8D7D2", 18);
  text(slide, "User · start · end\nrequested length", 100, 458, 180, 66, { size: 21, bold: true, color: C.ink, align: "center" });
  text(slide, "Frozen Rollout\n(no retraining)", 420, 458, 195, 66, { size: 23, bold: true, color: C.white, align: "center" });
  text(slide, "Ordered loop-free\nitinerary", 756, 458, 175, 66, { size: 22, bold: true, color: C.tealDark, align: "center" });
  connector(slide, q, f, { fromSide: "right", toSide: "left", color: C.orange });
  connector(slide, f, i, { fromSide: "right", toSide: "left", color: C.teal });
  const evalItinerary = box(slide, 988, 424, 218, 144, C.white, C.light, 18);
  text(slide, "NYC: internal comparison\n\nFlickr Benchmark: literature bridge", 1006, 442, 182, 104, { size: 18, bold: true, color: C.navy, align: "center" });
  connector(slide, i, evalItinerary, { fromSide: "right", toSide: "left", color: C.tealDark });
  box(slide, 360, 590, 560, 42, C.navy, C.navy, 16);
  text(slide, "Never mix scores across tasks, datasets, protocols, or metrics", 382, 597, 516, 28, { size: 18, bold: true, color: C.white, align: "center" });
  source(slide, "Source: Thesis Sections 3.2 and 3.6. Two tasks are evaluated separately.");
  setTalkNotes(slide, 3);
  return slide;
}

function buildSlide5(presentation) {
  const slide = presentation.slides.add();
  header(slide, "The engine combines places, context, and user preferences", 5, false, "4 · IMPLEMENTATION AND EVALUATION");

  const seq = box(slide, 70, 208, 170, 100, C.pale, C.light, 18);
  const graph = box(slide, 294, 178, 210, 112, C.tealPale, "#A8D7D2", 18);
  const context = box(slide, 294, 330, 210, 100, C.orangePale, "#F3C49D", 18);
  const user = box(slide, 888, 470, 160, 92, C.pale, C.light, 18);
  const gru = box(slide, 596, 260, 220, 150, C.navy, C.navy, 22);
  const fusion = box(slide, 888, 260, 160, 150, C.tealDark, C.tealDark, 22);
  const head = box(slide, 1090, 260, 140, 150, C.orange, C.orange, 22);

  text(slide, "Visited POI\nsequence", 92, 229, 126, 58, { size: 22, bold: true, align: "center", color: C.navy });
  text(slide, "Hybrid POI graph\n2-layer GCN", 318, 203, 162, 60, { size: 22, bold: true, align: "center", color: C.tealDark });
  text(slide, "Context\nΔd · Δt", 336, 348, 126, 60, { size: 22, bold: true, align: "center", color: C.ink });
  text(slide, "User embedding\n64 dimensions", 902, 483, 132, 55, { size: 18, bold: true, align: "center", color: C.navy });
  text(slide, "GRU\nsequence state", 620, 298, 172, 74, { size: 27, bold: true, align: "center", color: C.white });
  text(slide, "Fuse\nuser", 914, 301, 108, 70, { size: 25, bold: true, align: "center", color: C.white });
  text(slide, "MLP +\nsoftmax", 1108, 300, 104, 70, { size: 23, bold: true, align: "center", color: C.white });

  connector(slide, seq, graph, { fromSide: "right", toSide: "left", color: C.tealDark });
  connector(slide, graph, gru, { fromSide: "right", toSide: "left", color: C.tealDark });
  connector(slide, context, gru, { fromSide: "right", toSide: "left", color: C.orange });
  connector(slide, gru, fusion, { fromSide: "right", toSide: "left", color: C.navy2 });
  connector(slide, user, fusion, { fromSide: "top", toSide: "bottom", color: C.muted });
  connector(slide, fusion, head, { fromSide: "right", toSide: "left", color: C.tealDark });

  box(slide, 572, 478, 290, 78, C.white, C.light, 16);
  text(slide, "~10⁵ prefix → next-POI\ntraining examples", 594, 492, 246, 48, { size: 20, bold: true, align: "center", color: C.navy });
  box(slide, 1060, 468, 185, 90, C.white, C.light, 16);
  text(slide, "Scores all\n~5,100 POIs", 1080, 488, 145, 48, { size: 20, bold: true, align: "center", color: C.orange });
  source(slide, "Source: Thesis Sections 3.3-3.4. Implemented context is distance and elapsed time.");
  setTalkNotes(slide, 4);
  return slide;
}

function buildSlide7(presentation) {
  const slide = presentation.slides.add();
  header(slide, "The engine is competitive, but not state of the art", 7, false, "4 · IMPLEMENTATION AND EVALUATION");

  box(slide, 70, 174, 245, 420, C.navy, C.navy, 24);
  text(slide, "0.187", 92, 222, 200, 92, { size: 64, bold: true, color: C.white, align: "center" });
  text(slide, "HR@1", 92, 316, 200, 42, { size: 27, bold: true, color: C.tealPale, align: "center" });
  text(slide, "Exact next POI ranked first\n~1 time in 5 among ~5,100", 95, 380, 194, 86, { size: 19, color: "#D9E6EF", align: "center" });
  rect(slide, 108, 500, 168, 4, C.orange);
  text(slide, "HR@10 = 0.588", 96, 520, 192, 32, { size: 20, bold: true, color: C.white, align: "center" });

  const models = [
    ["LLM4POI", 0.340, C.navy2], ["STHGCN", 0.270, C.navy2], ["GETNext", 0.240, C.navy2],
    ["STAN", 0.220, C.navy2], ["OURS", 0.187, C.orange], ["STGCN", 0.180, C.tealDark], ["LSTM", 0.130, C.muted],
  ];
  const x0 = 490;
  const maxW = 650;
  const maxV = 0.36;
  for (let i = 0; i < models.length; i += 1) {
    const [name, value, color] = models[i];
    const y = 178 + i * 59;
    text(slide, name, 338, y + 5, 132, 28, { size: name === "OURS" ? 20 : 18, bold: name === "OURS", color: name === "OURS" ? C.orange : C.ink, align: "right" });
    rect(slide, x0, y, maxW, 34, "#E7ECEF");
    rect(slide, x0, y, (value / maxV) * maxW, 34, color);
    text(slide, value.toFixed(3), x0 + (value / maxV) * maxW + 10, y + 3, 70, 28, { size: 17, bold: true, color: C.ink });
  }
  text(slide, "Higher is better · full vocabulary · Foursquare NYC", 490, 604, 650, 28, { size: 16, color: C.muted, align: "center" });
  source(slide, "Source: Thesis Table 3.2 and Figure 3.2. Published comparison values use the same NYC full-vocabulary HR@1 protocol.");
  setTalkNotes(slide, 6);
  return slide;
}

function buildSlide6(presentation) {
  const slide = presentation.slides.add();
  header(slide, "Frozen Rollout turns next-POI scores into a loop-free route", 6, false, "4 · IMPLEMENTATION AND EVALUATION");

  const steps = [
    ["1", "QUERY", "user · start · end · K", C.navy],
    ["2", "SCORE", "run frozen engine", C.tealDark],
    ["3", "MASK", "visited + early end", C.orange],
    ["4", "SELECT", "best valid candidate", C.tealDark],
    ["5", "REPEAT", "until length K", C.navy],
  ];
  const shapes = [];
  for (let i = 0; i < steps.length; i += 1) {
    const [n, label, detail, color] = steps[i];
    const x = 66 + i * 238;
    const b = box(slide, x, 190, 196, 146, i === 2 ? C.orangePale : C.white, i === 2 ? "#F3C49D" : C.light, 20);
    circle(slide, x + 30, 220, 34, color, n, { size: 17, line: color, lineWidth: 0, shadow: "shadow-none" });
    text(slide, label, x + 58, 202, 118, 32, { size: 20, bold: true, color });
    text(slide, detail, x + 20, 252, 156, 56, { size: 18, color: C.ink, align: "center" });
    shapes.push(b);
  }
  for (let i = 0; i < shapes.length - 1; i += 1) connector(slide, shapes[i], shapes[i + 1], { fromSide: "right", toSide: "left", color: C.muted, width: 3 });

  text(slide, "The route grows one valid stop at a time", 68, 382, 520, 36, { size: 24, bold: true, color: C.navy });
  const nodes = [
    circle(slide, 170, 520, 54, C.navy, "S"),
    circle(slide, 355, 468, 54, C.teal, "1"),
    circle(slide, 565, 528, 54, C.teal, "2"),
    circle(slide, 790, 455, 54, C.orange, "3"),
    circle(slide, 1055, 520, 54, C.navy, "E"),
  ];
  for (let i = 0; i < nodes.length - 1; i += 1) connector(slide, nodes[i], nodes[i + 1], { color: C.tealDark, width: 5 });
  text(slide, "No revisits", 105, 572, 130, 28, { size: 18, bold: true, color: C.tealDark, align: "center" });
  text(slide, "Destination reserved for the last step", 880, 572, 330, 28, { size: 18, bold: true, color: C.orange, align: "center" });
  box(slide, 460, 592, 350, 42, C.navy, C.navy, 16);
  text(slide, "Same checkpoint · zero extra training", 482, 599, 306, 28, { size: 18, bold: true, color: C.white, align: "center" });
  source(slide, "Source: Thesis Section 3.5. Decode-time constraints: loop-free, fixed endpoints, requested length.");
  setTalkNotes(slide, 5);
  return slide;
}

function buildSlide8(presentation) {
  const slide = presentation.slides.add();
  header(slide, "Frozen Rollout wins; dense supervision helps explain why", 8, false, "4 · IMPLEMENTATION AND EVALUATION");

  box(slide, 70, 176, 515, 395, C.tealPale, "#A8D7D2", 24);
  box(slide, 695, 176, 515, 395, C.orangePale, "#F3C49D", 24);
  text(slide, "FROZEN ROLLOUT", 106, 205, 440, 32, { size: 22, bold: true, color: C.tealDark, align: "center" });
  text(slide, "TRAINED POINTER", 731, 205, 440, 32, { size: 22, bold: true, color: C.orange, align: "center" });

  text(slide, "~10⁵", 145, 268, 220, 82, { size: 58, bold: true, color: C.tealDark, align: "center" });
  text(slide, "prefix → next-POI examples", 110, 346, 300, 34, { size: 20, bold: true, color: C.ink, align: "center" });
  text(slide, "Reuses a converged engine", 110, 398, 300, 34, { size: 18, color: C.muted, align: "center" });
  text(slide, "0.289", 145, 448, 220, 70, { size: 52, bold: true, color: C.navy, align: "center" });
  text(slide, "pairs-F1", 178, 518, 154, 28, { size: 19, bold: true, color: C.tealDark, align: "center" });

  text(slide, "1", 780, 268, 220, 82, { size: 58, bold: true, color: C.orange, align: "center" });
  text(slide, "whole trajectory per session", 735, 346, 310, 34, { size: 20, bold: true, color: C.ink, align: "center" });
  text(slide, "Trains end-to-end from scratch", 735, 398, 310, 34, { size: 18, color: C.muted, align: "center" });
  text(slide, "0.259", 780, 448, 220, 70, { size: 52, bold: true, color: C.navy, align: "center" });
  text(slide, "pairs-F1", 813, 518, 154, 28, { size: 19, bold: true, color: C.orange, align: "center" });

  circle(slide, 640, 372, 84, C.navy, ">", { size: 42, line: C.white, lineWidth: 4 });
  box(slide, 407, 595, 466, 42, C.navy, C.navy, 16);
  text(slide, "NYC · n = 2,880 · length >= 3 · order-aware pairs-F1", 430, 602, 420, 27, { size: 17, bold: true, color: C.white, align: "center" });
  source(slide, "Source: Thesis Table 3.3. Beam search changes Frozen Rollout only from 0.289 to 0.290.");
  setTalkNotes(slide, 7);
  return slide;
}

function buildSlide3(presentation) {
  const slide = presentation.slides.add();
  header(slide, "Existing methods trade off personalization and feasibility", 3, false, "2 · STATE OF THE ART");

  const cards = [
    { x: 70, fill: C.orangePale, line: "#F3C49D", color: C.orange, number: "1", title: "ROUTE OPTIMIZATION", methods: "OP · TTDP", strength: "Constraints by construction", limit: "Simple utility; weak personalization" },
    { x: 460, fill: C.pale, line: C.light, color: C.navy2, number: "2", title: "PERSONALIZED + CONTEXT", methods: "CF · hybrid · CARS", strength: "Learns user and context relevance", limit: "Usually ranks POIs, not routes" },
    { x: 850, fill: C.tealPale, line: "#A8D7D2", color: C.tealDark, number: "3", title: "DEEP SEQUENTIAL MODELS", methods: "RNN · Transformer · GNN", strength: "Learns mobility transitions", limit: "Local objective; greedy route" },
  ];
  for (const card of cards) {
    box(slide, card.x, 190, 340, 300, card.fill, card.line, 22);
    circle(slide, card.x + 42, 226, 40, card.color, card.number, { size: 19, line: C.white, lineWidth: 2, shadow: "shadow-none" });
    text(slide, card.title, card.x + 72, 207, 242, 36, { size: 18, bold: true, color: card.color });
    text(slide, card.methods, card.x + 28, 270, 284, 40, { size: 26, bold: true, color: C.navy, align: "center" });
    text(slide, "STRENGTH", card.x + 28, 336, 284, 24, { size: 13, bold: true, color: C.tealDark, align: "center" });
    text(slide, card.strength, card.x + 30, 363, 280, 48, { size: 18, bold: true, color: C.ink, align: "center" });
    text(slide, "LIMIT", card.x + 28, 425, 284, 24, { size: 13, bold: true, color: C.orange, align: "center" });
    text(slide, card.limit, card.x + 30, 450, 280, 32, { size: 17, color: C.ink, align: "center" });
  }

  box(slide, 70, 525, 1120, 102, C.navy, C.navy, 20);
  text(slide, "RESEARCH GAP", 96, 542, 190, 28, { size: 16, bold: true, color: C.orange });
  text(slide, "Next-POI accuracy ≠ itinerary quality - and evaluation remains fragmented.", 285, 540, 875, 40, { size: 24, bold: true, color: C.white, align: "center" });
  text(slide, "This thesis connects learned relevance to constrained route generation, then evaluates each task separately.", 116, 584, 1030, 28, { size: 18, color: "#D6E4EE", align: "center" });
  source(slide, "Source: Thesis Chapter 2, Sections 2.3-2.8; especially Table 2.3 and Research Gaps 3 and 5.");
  setTalkNotes(slide, 2);
  return slide;
}

function buildSlide9Legacy(presentation) {
  const slide = presentation.slides.add();
  header(slide, "The benchmark confirms our position - and exposes cold-start", 9);

  text(slide, "FLICKR BENCHMARK · PAIRS-F1 ON THE PUBLISHED SCALE", 70, 170, 690, 28, { size: 17, bold: true, color: C.navy });
  const axisX = 250;
  const axisW = 520;
  const rows = [
    ["Markov family", 0.23, 0.59, C.tealDark],
    ["Trained Pointer", 0.31, 0.49, C.orange],
    ["SelfTrip / AR-Trip", 0.78, 0.85, C.navy2],
  ];
  line(slide, axisX, 510, axisX + axisW, 510, C.muted, 2);
  for (let tick = 0; tick <= 10; tick += 2) {
    const x = axisX + (tick / 10) * axisW;
    line(slide, x, 506, x, 518, C.muted, 1);
    text(slide, (tick / 10).toFixed(1), x - 20, 522, 40, 22, { size: 13, color: C.muted, align: "center" });
  }
  for (let i = 0; i < rows.length; i += 1) {
    const [label, low, high, color] = rows[i];
    const y = 230 + i * 104;
    text(slide, label, 70, y, 160, 34, { size: 19, bold: true, color: C.ink, align: "right" });
    rect(slide, axisX, y + 2, axisW, 30, "#E7ECEF");
    rect(slide, axisX + low * axisW, y + 2, (high - low) * axisW, 30, color);
    text(slide, `${low.toFixed(2)} - ${high.toFixed(2)}`, axisX + low * axisW, y + 38, Math.max(105, (high - low) * axisW), 24, { size: 15, bold: true, color, align: "center" });
  }
  box(slide, 82, 575, 676, 48, C.navy, C.navy, 16);
  text(slide, "Our classical methods reach the published scale; neural SOTA remains clearly ahead", 104, 582, 632, 32, { size: 17, bold: true, color: C.white, align: "center" });

  text(slide, "USER EMBEDDING ABLATION", 860, 170, 330, 28, { size: 17, bold: true, color: C.navy });
  box(slide, 842, 214, 360, 352, C.white, C.light, 22);
  const deltas = [["Glasgow", 0.038, C.green], ["Osaka", -0.007, C.orange], ["Toronto", -0.017, C.red]];
  const baseY = 405;
  line(slide, 890, baseY, 1155, baseY, C.muted, 2);
  for (let i = 0; i < deltas.length; i += 1) {
    const [city, value, color] = deltas[i];
    const x = 920 + i * 92;
    const h = Math.abs(value) / 0.04 * 120;
    rect(slide, x, value >= 0 ? baseY - h : baseY, 52, h, color);
    text(slide, `${value >= 0 ? "+" : ""}${value.toFixed(3)}`, x - 14, value >= 0 ? baseY - h + 8 : baseY + h + 5, 80, 24, { size: 16, bold: true, color: value >= 0 ? C.white : color, align: "center" });
    text(slide, city, x - 20, 500, 92, 24, { size: 14, color: C.ink, align: "center" });
  }
  text(slide, "Personalization helps only when enough user history survives leave-one-out", 875, 235, 294, 56, { size: 19, bold: true, color: C.navy, align: "center" });
  text(slide, "Separate 30-epoch ablation", 884, 535, 278, 22, { size: 13, italic: true, color: C.muted, align: "center" });
  source(slide, "Source: Thesis Tables 3.6-3.8 and Figure 3.5. The ablation is separate from the main Trained Pointer run.");
  setTalkNotes(slide, 8);
  return slide;
}

function buildSlide9(presentation) {
  const slide = presentation.slides.add();
  header(slide, "Flickr validates the protocol - and shows the remaining gap", 9, false, "4 · IMPLEMENTATION AND EVALUATION");

  box(slide, 70, 180, 330, 430, C.navy, C.navy, 22);
  text(slide, "FLICKR BENCHMARK", 98, 205, 274, 28, { size: 18, bold: true, color: C.orange, align: "center" });
  text(slide, "27 - 88 POIs per city", 98, 246, 274, 34, { size: 23, bold: true, color: C.white, align: "center" });
  text(slide, "PROTOCOL CHECK", 98, 302, 274, 24, { size: 14, bold: true, color: C.tealPale, align: "center" });
  text(slide, "Leave one trajectory out\nFirst + last POI + length\nLength >= 3 · loop-free · pairs-F1", 102, 338, 266, 92, { size: 17, color: "#D6E4EE", align: "center" });
  text(slide, "0.021", 98, 442, 274, 70, { size: 52, bold: true, color: C.white, align: "center" });
  text(slide, "maximum Random reproduction gap", 100, 510, 270, 36, { size: 16, color: "#D6E4EE", align: "center" });
  box(slide, 104, 558, 262, 34, C.tealDark, C.tealDark, 12);
  text(slide, "Protocol faithfully reproduced", 116, 563, 238, 24, { size: 16, bold: true, color: C.white, align: "center" });

  box(slide, 435, 180, 775, 270, C.white, C.light, 22);
  text(slide, "PAIRS-F1 ON THE PUBLISHED SCALE", 470, 204, 700, 28, { size: 18, bold: true, color: C.navy });
  const axisX = 680;
  const axisW = 450;
  const rows = [
    ["Classical baselines", 0.23, 0.59, C.tealDark],
    ["Trained Pointer", 0.31, 0.49, C.orange],
    ["SelfTrip / AR-Trip", 0.78, 0.85, C.navy2],
  ];
  for (let i = 0; i < rows.length; i += 1) {
    const [label, low, high, color] = rows[i];
    const y = 258 + i * 62;
    text(slide, label, 470, y - 2, 185, 28, { size: 18, bold: true, color: C.ink, align: "right" });
    rect(slide, axisX, y, axisW, 26, "#E7ECEF");
    rect(slide, axisX + low * axisW, y, (high - low) * axisW, 26, color);
    text(slide, `${low.toFixed(2)} - ${high.toFixed(2)}`, axisX + low * axisW, y + 30, Math.max(105, (high - low) * axisW), 22, { size: 14, bold: true, color, align: "center" });
  }
  text(slide, "Our reproduced baselines reach the published scale; neural SOTA remains clearly ahead.", 475, 420, 690, 24, { size: 16, bold: true, color: C.navy, align: "center" });

  box(slide, 435, 475, 775, 135, C.pale, C.light, 20);
  text(slide, "SEPARATE USER-EMBEDDING ABLATION", 465, 494, 370, 24, { size: 16, bold: true, color: C.navy });
  const deltas = [
    ["Glasgow", "+0.038", C.green],
    ["Osaka", "-0.007", C.orange],
    ["Toronto", "-0.017", C.red],
  ];
  for (let i = 0; i < deltas.length; i += 1) {
    const [city, value, color] = deltas[i];
    const x = 470 + i * 150;
    text(slide, value, x, 530, 132, 32, { size: 23, bold: true, color, align: "center" });
    text(slide, city, x, 562, 132, 22, { size: 14, color: C.ink, align: "center" });
  }
  text(slide, "Sparse leave-one-out history creates cold-start.", 920, 525, 245, 52, { size: 17, bold: true, color: C.ink, align: "center" });
  source(slide, "Source: Thesis Tables 3.4-3.8 and Figures 3.4-3.5. NYC and Flickr scores are never compared directly.");
  setTalkNotes(slide, 8);
  return slide;
}

function buildSlide10(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.navy;
  text(slide, "5 · CONCLUSION", 68, 46, 520, 30, { size: 16, bold: true, color: C.orange });
  text(slide, "Three results to remember", 68, 92, 900, 56, { size: 42, bold: true, color: C.white });

  const nodes = [
    circle(slide, 170, 360, 78, C.tealDark, "1", { size: 30 }),
    circle(slide, 640, 360, 78, C.orange, "2", { size: 30 }),
    circle(slide, 1110, 360, 78, C.teal, "3", { size: 30 }),
  ];
  connector(slide, nodes[0], nodes[1], { color: "#5A7891", width: 5 });
  connector(slide, nodes[1], nodes[2], { color: "#5A7891", width: 5 });

  text(slide, "0.187 HR@1", 70, 214, 300, 60, { size: 37, bold: true, color: C.white, align: "center" });
  text(slide, "honest next-POI engine\n~5,100 NYC candidates", 70, 468, 300, 62, { size: 21, color: "#D6E4EE", align: "center" });

  text(slide, "0.289 > 0.259", 490, 214, 300, 60, { size: 37, bold: true, color: C.white, align: "center" });
  text(slide, "Frozen Rollout beats\nthe Trained Pointer", 490, 468, 300, 62, { size: 21, color: "#D6E4EE", align: "center" });

  text(slide, "max gap 0.021", 910, 214, 300, 60, { size: 37, bold: true, color: C.white, align: "center" });
  text(slide, "Flickr Benchmark reproduces\nChen 2016 Random", 910, 468, 300, 62, { size: 21, color: "#D6E4EE", align: "center" });

  box(slide, 220, 586, 840, 50, "#153B5B", "#153B5B", 18);
  text(slide, "Compare only when dataset + protocol + metric are identical", 250, 596, 780, 30, { size: 21, bold: true, color: C.white, align: "center" });
  text(slide, "Questions", 1030, 664, 180, 26, { size: 18, bold: true, color: C.orange, align: "right" });
  text(slide, "Ayman Naaimi · ENSAM Meknes", 68, 664, 420, 26, { size: 13, color: "#AFC0CD" });
  setTalkNotes(slide, 9);
  return slide;
}

function addTable(slide, values, x, y, w, h, options = {}) {
  const columns = values[0].length;
  const weights = options.weights || new Array(columns).fill(1);
  const table = slide.tables.add({
    rows: values.length,
    columns,
    left: x,
    top: y,
    width: w,
    height: h,
    columnTracks: weights.map((value) => ({ mode: "fr", value })),
    values,
  });
  table.borders.assign({ style: "solid", fill: options.border || C.light, width: 1 });
  for (let r = 0; r < values.length; r += 1) {
    for (let c = 0; c < columns; c += 1) {
      const cell = table.getCell(r, c);
      cell.fill = r === 0 ? (options.headerFill || C.navy) : (r % 2 === 0 ? C.pale : C.white);
      cell.text.style = {
        typeface: FONT,
        fontSize: options.fontSize || 15,
        bold: r === 0 || (options.boldFirst && c === 0),
        color: r === 0 ? C.white : C.ink,
        alignment: c === 0 ? "left" : "center",
        verticalAlignment: "middle",
        autoFit: "shrinkText",
      };
    }
  }
  return table;
}

function backupNotes(title, use) {
  return [
    `BACKUP SLIDE: ${title}`,
    `USE WHEN: ${use}`,
    "",
    "Keep the answer short, point to the requested evidence, and return to the closing slide.",
  ].join("\n");
}

function buildSlide11(presentation) {
  const slide = presentation.slides.add();
  header(slide, "Data preparation and evaluation protocols", 11, true);

  text(slide, "PHASE 1 · FOURSQUARE NYC", 70, 166, 500, 30, { size: 19, bold: true, color: C.navy });
  const phase1 = [
    ["Raw data", "~227K check-ins · ~1.08K users · ~38K venues"],
    ["Filter", "Iterative user/POI count >= 10"],
    ["Final catalog", "~5,100 POIs · ~1,000 users"],
    ["Sessions", "Split after >24 h silence"],
    ["Split", "Chronological per user: 70 / 10 / 20"],
    ["Evaluation", "Full vocabulary · no sampled negatives"],
  ];
  for (let i = 0; i < phase1.length; i += 1) {
    const y = 210 + i * 62;
    circle(slide, 91, y + 22, 28, i === phase1.length - 1 ? C.orange : C.tealDark, String(i + 1), { size: 14, lineWidth: 0, shadow: "shadow-none" });
    text(slide, phase1[i][0], 118, y, 128, 28, { size: 17, bold: true, color: C.navy });
    text(slide, phase1[i][1], 244, y, 338, 45, { size: 16, color: C.ink });
  }

  text(slide, "PHASE 2 · FLICKR BENCHMARK", 650, 166, 500, 30, { size: 19, bold: true, color: C.orange });
  addTable(slide, [
    ["City", "POIs", "Users", "Traj.", "Eval. >=3"],
    ["Toronto", "29", "1,395", "6,057", "335"],
    ["Osaka", "27", "450", "1,115", "47"],
    ["Glasgow", "27", "601", "2,227", "112"],
    ["Edinburgh", "28", "1,454", "5,028", "634"],
    ["Melbourne", "88", "1,000", "5,106", "442"],
  ], 650, 210, 560, 260, { weights: [1.7, 0.8, 1.0, 1.0, 1.1], fontSize: 15, headerFill: C.orange, boldFirst: true });
  box(slide, 650, 500, 560, 106, C.orangePale, "#F3C49D", 18);
  text(slide, "Chen-2016 protocol", 675, 515, 210, 26, { size: 18, bold: true, color: C.orange });
  text(slide, "Leave one trajectory out · first + last POI + length given\nlength >= 3 · loop-free · pairs-F1", 675, 548, 505, 48, { size: 17, color: C.ink });
  source(slide, "Source: Thesis Sections 3.3 and 3.6; Table 3.4.");
  slide.speakerNotes.textFrame.setText(backupNotes("Data preparation and evaluation protocols", "The jury asks about filtering, leakage, dataset sizes, or the exact Chen-2016 protocol."));
  slide.speakerNotes.setVisible(true);
  return slide;
}

function buildSlide12(presentation) {
  const slide = presentation.slides.add();
  header(slide, "Metric definitions and complete result tables", 12, true);

  text(slide, "NEXT-POI · NYC", 66, 160, 290, 26, { size: 18, bold: true, color: C.navy });
  addTable(slide, [
    ["Metric", "Score"],
    ["HR@1", "0.187"], ["HR@5", "0.476"], ["HR@10", "0.588"],
    ["NDCG@5", "0.339"], ["NDCG@10", "0.375"], ["MRR", "0.316"],
  ], 66, 194, 270, 264, { weights: [1.5, 1], fontSize: 15, boldFirst: true });

  text(slide, "ITINERARY · NYC", 66, 486, 290, 26, { size: 18, bold: true, color: C.orange });
  addTable(slide, [
    ["Method", "pairs", "set", "exact"],
    ["Frozen Rollout · greedy", ".289", ".609", ".054"],
    ["Frozen Rollout · beam 3", ".290", ".610", ".057"],
    ["Trained Pointer · v1", ".259", ".578", ".043"],
    ["Trained Pointer · v2 +ctx", ".261", "-", "-"],
  ], 66, 520, 470, 124, { weights: [2.2, 0.8, 0.8, 0.8], fontSize: 14, headerFill: C.orange, boldFirst: true });

  text(slide, "FLICKR BENCHMARK · PAIRS-F1", 370, 160, 600, 26, { size: 18, bold: true, color: C.tealDark });
  addTable(slide, [
    ["Method", "Toronto", "Osaka", "Glasgow", "Edin.", "Melb."],
    ["Random", ".298", ".301", ".301", ".270", ".227"],
    ["PoiPopularity", ".443", ".413", ".510", ".439", ".320"],
    ["Markov", ".504", ".421", ".587", ".449", ".333"],
    ["MarkovPath", ".528", ".398", ".543", ".452", ".346"],
    ["Trained Pointer · beam 3", ".431", ".399", ".489", ".414", ".312"],
  ], 370, 194, 840, 264, { weights: [2.0, 1, 1, 1, 1, 1], fontSize: 15, headerFill: C.tealDark, boldFirst: true });

  box(slide, 578, 500, 632, 144, C.navy, C.navy, 18);
  text(slide, "METRIC BOUNDARY", 604, 514, 190, 24, { size: 16, bold: true, color: C.orange });
  text(slide, "HR@k / NDCG / MRR rank one next POI.\nset-F1 measures recovered places. pairs-F1 measures order.\nThese values never share one leaderboard.", 604, 548, 578, 78, { size: 17, color: C.white });
  source(slide, "Source: Thesis Tables 3.2, 3.3 and 3.6. Dashes mean unreported, not zero.");
  slide.speakerNotes.textFrame.setText(backupNotes("Metric definitions and complete result tables", "The jury requests an exact number, asks what pairs-F1 measures, or challenges metric comparability."));
  slide.speakerNotes.setVisible(true);
  return slide;
}

function buildSlide13(presentation) {
  const slide = presentation.slides.add();
  header(slide, "Detailed architecture, dimensions, and training controls", 13, true);

  const g = box(slide, 70, 210, 200, 104, C.tealPale, "#A8D7D2", 18);
  const c = box(slide, 70, 356, 200, 104, C.orangePale, "#F3C49D", 18);
  const u = box(slide, 70, 502, 200, 84, C.pale, C.light, 18);
  const gru = box(slide, 390, 280, 220, 150, C.navy, C.navy, 22);
  const out = box(slide, 730, 280, 220, 150, C.tealDark, C.tealDark, 22);
  text(slide, "2-layer GCN\nPOI dim = 128", 94, 230, 152, 60, { size: 21, bold: true, color: C.tealDark, align: "center" });
  text(slide, "Δd · Δt MLPs\ncontext dim = 32", 94, 376, 152, 60, { size: 21, bold: true, color: C.ink, align: "center" });
  text(slide, "User embedding\ndim = 64", 94, 514, 152, 50, { size: 20, bold: true, color: C.navy, align: "center" });
  text(slide, "GRU\nhidden dim = 128", 418, 312, 164, 80, { size: 24, bold: true, color: C.white, align: "center" });
  text(slide, "MLP + softmax\n~5,100 classes", 758, 312, 164, 80, { size: 23, bold: true, color: C.white, align: "center" });
  connector(slide, g, gru, { color: C.tealDark });
  connector(slide, c, gru, { color: C.orange });
  connector(slide, gru, out, { color: C.navy2 });
  connector(slide, u, out, { color: C.muted, kind: "elbow", fromSide: "right", toSide: "bottom" });

  box(slide, 1005, 194, 205, 390, C.white, C.light, 20);
  text(slide, "IMPLEMENTED SETTINGS", 1027, 216, 162, 44, { size: 16, bold: true, color: C.orange, align: "center" });
  text(slide, "Graph\nco-visits >= 3\n+ 10 nearest POIs\n\nContext bounds\nΔd: 0-100 km\nΔt: 0-24 h\n\nTraining\ncross-entropy\nAdam · clipping · dropout\nearly stop on HR@10", 1032, 272, 152, 288, { size: 17, color: C.ink, align: "center", valign: "top" });
  box(slide, 340, 510, 620, 76, C.navy, C.navy, 18);
  text(slide, "Training-only co-visitation edges prevent validation and test leakage.", 368, 526, 564, 44, { size: 18, bold: true, color: C.white, align: "center" });
  source(slide, "Source: Thesis Sections 3.3-3.4. Co-visitation edges are computed from training data only.");
  slide.speakerNotes.textFrame.setText(backupNotes("Detailed architecture, dimensions, and training controls", "The jury asks for dimensions, graph construction, context bounds, training objective, or leakage controls."));
  slide.speakerNotes.setVisible(true);
  return slide;
}

function buildSlide14(presentation) {
  const slide = presentation.slides.add();
  header(slide, "Why the Trained Pointer loses - limitations and next steps", 14, true);

  text(slide, "TRAINING CONTROLS", 70, 164, 340, 28, { size: 18, bold: true, color: C.tealDark });
  box(slide, 70, 206, 350, 352, C.tealPale, "#A8D7D2", 22);
  text(slide, "0.259 → 0.261", 98, 234, 294, 58, { size: 39, bold: true, color: C.navy, align: "center" });
  text(slide, "Adding context recovers only +0.002", 98, 298, 294, 38, { size: 18, bold: true, color: C.tealDark, align: "center" });
  text(slide, "✓ validation pairs-F1 reaches a real peak\n✓ early stopping applied\n✓ context-aware variant tested\n✓ Trained Pointer also trails Markov on the Flickr Benchmark\n\nConclusion: a genuine negative result for this tested setup, not a universal claim about all itinerary models.", 104, 354, 282, 174, { size: 17, color: C.ink, valign: "top" });

  text(slide, "MAIN LIMITATIONS", 462, 164, 300, 28, { size: 18, bold: true, color: C.orange });
  box(slide, 462, 206, 350, 352, C.orangePale, "#F3C49D", 22);
  text(slide, "• below neural / LLM SOTA\n• scorer remains locally myopic\n• context only uses Δd and Δt\n• decode-time Δt is assumed\n• no opening hours or time windows\n• one next-POI city\n• offline evaluation only\n• no deployment or user study", 500, 242, 274, 278, { size: 18, color: C.ink, valign: "top" });

  text(slide, "NEXT STEPS", 852, 164, 300, 28, { size: 18, bold: true, color: C.navy });
  box(slide, 852, 206, 358, 352, C.pale, C.light, 22);
  text(slide, "• weather and time-of-day context\n• persona/content cold-start encoder\n• travel cost and opening hours\n• schedule-aware constrained decoding\n• pre-training and trajectory augmentation\n• multi-city validation\n• user study", 888, 242, 286, 230, { size: 18, color: C.ink, valign: "top" });
  box(slide, 882, 490, 298, 48, C.navy, C.navy, 14);
  text(slide, "Orienteering / ILS: scoped, NOT implemented", 896, 499, 270, 30, { size: 16, bold: true, color: C.white, align: "center" });
  source(slide, "Source: Thesis Sections 3.7.5 and General Conclusion. Reader-facing method names are used throughout.");
  slide.speakerNotes.textFrame.setText(backupNotes("Why the Trained Pointer loses - limitations and next steps", "The jury challenges the negative result, asks whether training failed, or asks about limitations and future work."));
  slide.speakerNotes.setVisible(true);
  return slide;
}

async function build() {
  const args = argsFrom(process.argv.slice(2));
  const output = path.resolve(args.output || path.join(__dirname, "PFE_AYMAN_NAAIMI_10MIN_EN_COORDINATOR.pptx"));
  const previewDir = path.resolve(args.previewDir || path.join(__dirname, "preview-10min-en"));
  const layoutDir = path.resolve(args.layoutDir || path.join(__dirname, "layout-10min-en"));
  const montagePath = path.resolve(args.montage || path.join(previewDir, "montage.webp"));
  const notesOut = path.resolve(args.notesOut || path.join(previewDir, "speaker-notes.json"));
  const heroPath = path.resolve(args.hero || path.join(__dirname, "assets", "hero_city_route_en.png"));

  await Promise.all([
    fs.mkdir(path.dirname(output), { recursive: true }),
    fs.mkdir(previewDir, { recursive: true }),
    fs.mkdir(layoutDir, { recursive: true }),
  ]);

  const heroBytes = await readBytes(heroPath);
  const presentation = Presentation.create({ slideSize: { width: SLIDE_W, height: SLIDE_H } });

  buildSlide1(presentation, heroBytes);
  buildSlide2(presentation);
  buildSlide3(presentation);
  buildSlide4(presentation);
  buildSlide5(presentation);
  buildSlide6(presentation);
  buildSlide7(presentation);
  buildSlide8(presentation);
  buildSlide9(presentation);
  buildSlide10(presentation);
  buildSlide11(presentation);
  buildSlide12(presentation);
  buildSlide13(presentation);
  buildSlide14(presentation);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(previewDir, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1.5 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(layoutDir, `${stem}.json`), await layout.text(), "utf8");
  }

  await writeBlob(montagePath, await presentation.export({ format: "webp", montage: true, scale: 1 }));
  await fs.writeFile(notesOut, `${JSON.stringify(TALK, null, 2)}\n`, "utf8");
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(output);
  await fs.rm(`${output}.inspect.ndjson`, { force: true });
  console.log(JSON.stringify({ output, previewDir, layoutDir, montagePath, notesOut, slides: presentation.slides.items.length }, null, 2));
}

build().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
