"use strict";

// The interface only reads the local API. Recorded strings always enter the DOM
// through text nodes; prompts, source files and model output are never HTML.
const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const ICONS = {
  search: ["circle", {cx:10,cy:10,r:6}, "path", {d:"m14.5 14.5 4 4"}],
  refresh: ["path", {d:"M18 8a7 7 0 1 0 1 6M18 3v5h-5"}],
  shield: ["path", {d:"m12 3 7 3v5c0 5-7 9-7 9s-7-4-7-9V6l7-3Z M8.5 11.5l2.3 2.3 4.7-4.7"}],
  database: ["ellipse", {cx:12,cy:5,rx:7,ry:3}, "path", {d:"M5 5v13c0 4 14 4 14 0V5M5 11c0 4 14 4 14 0"}],
  layers: ["path", {d:"m12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5"}],
  folder: ["path", {d:"M3 7V5h7l2 2h9v12H3V7Z"}],
  task: ["rect", {x:5,y:4,width:14,height:17,rx:2}, "path", {d:"M9 4V2h6v2M9 10h6M9 14h6M9 18h4"}],
  questions: ["path", {d:"M4 4h16v12H9l-5 4V4ZM8 8h8M8 12h5"}],
  evolution: ["path", {d:"M4 16 9 11l4 3 7-9M15 5h5v5M4 5v15h16"}],
  input: ["rect", {x:4,y:4,width:16,height:16,rx:3}, "path", {d:"M8 9h8M8 13h8M8 17h4"}],
  manage: ["circle", {cx:12,cy:5,r:2}, "rect", {x:3,y:15,width:6,height:5,rx:1}, "rect", {x:15,y:15,width:6,height:5,rx:1}, "path", {d:"M12 7v4M6 15v-4h12v4"}],
  agentic: ["path", {d:"m12 3 8 5v8l-8 5-8-5V8l8-5ZM4 8l8 5 8-5M12 13v8M8 5.5l8 5"}],
  components: ["rect", {x:3,y:3,width:7,height:7,rx:1}, "rect", {x:14,y:3,width:7,height:7,rx:1}, "rect", {x:3,y:14,width:7,height:7,rx:1}, "path", {d:"M14 17.5h7M17.5 14v7"}],
  execution: ["circle", {cx:12,cy:12,r:9}, "path", {d:"m10 8 6 4-6 4V8Z"}],
  evaluation: ["path", {d:"M4 20V10M10 20V4M16 20v-7M22 20H2"}],
  policy_select: ["path", {d:"m3 4 9 7 9-7M3 4h18v16H3V4ZM8 16l3 3 5-6"}],
  policy_edit: ["path", {d:"m15 4 5 5M4 15 16 3a2 2 0 0 1 5 5L9 20l-6 1 1-6ZM4 15l5 5"}],
  validation: ["path", {d:"m12 3 8 4v5c0 5-8 9-8 9s-8-4-8-9V7l8-4ZM8 12l3 3 5-6"}],
  inheritance: ["path", {d:"M5 4v6c0 4 14 0 14 4v6M5 10v10M16 17l3 3 3-3"}, "circle", {cx:5,cy:3,r:2}, "circle", {cx:5,cy:21,r:2}],
  cursor: ["path", {d:"m5 3 14 9-7 1-3 7-4-17Z"}],
  "arrow-right": ["path", {d:"M4 12h16M14 6l6 6-6 6"}],
  "chevron-right": ["path", {d:"m9 5 7 7-7 7"}],
  "chevron-left": ["path", {d:"m15 5-7 7 7 7"}],
  copy: ["rect", {x:8,y:8,width:12,height:13,rx:2}, "path", {d:"M15 8V3H3v13h5"}],
  info: ["circle", {cx:12,cy:12,r:9}, "path", {d:"M12 11v6M12 7v.1"}],
  code: ["path", {d:"m8 6-6 6 6 6M16 6l6 6-6 6M14 3l-4 18"}],
  clock: ["circle", {cx:12,cy:12,r:9}, "path", {d:"M12 6v6l4 2"}],
  spark: ["path", {d:"m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z"}],
};

function icon(name, extraClass = "") {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", `icon ${extraClass}`.trim());
  svg.setAttribute("aria-hidden", "true");
  const shape = ICONS[name] || ICONS.info;
  for (let i = 0; i < shape.length; i += 2) {
    const child = document.createElementNS(SVG_NS, shape[i]);
    for (const [key, val] of Object.entries(shape[i + 1])) child.setAttribute(key, val);
    svg.append(child);
  }
  return svg;
}

function el(tag, className, ...children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  for (const child of children.flat()) {
    if (child !== undefined && child !== null) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}
function clear(node, ...children) { node.replaceChildren(...children.flat().filter(Boolean)); }
function button(className, children, onClick, label) {
  const node = el("button", className, children);
  node.type = "button";
  if (label) node.setAttribute("aria-label", label);
  node.addEventListener("click", onClick);
  return node;
}
const present = (v) => v !== null && v !== undefined && v !== "";
const nonempty = (v) => present(v) && (typeof v !== "object" || Object.keys(v).length > 0);
const text = (v) => !present(v) ? "此运行未记录" : typeof v === "string" ? v : JSON.stringify(v, null, 2);
const array = (v) => Array.isArray(v) ? v : [];
const shortHash = (v) => present(v) ? String(v).slice(0, 10) : "未记录";
const number = (v) => present(v) && Number.isFinite(Number(v)) ? Number(v).toLocaleString("zh-CN") : "—";
const duration = (v) => present(v) && Number.isFinite(Number(v)) ? (Number(v) < 1 ? `${(Number(v) * 1000).toFixed(0)} ms` : `${Number(v).toFixed(2)} s`) : "耗时未记录";

const FLOW = [
  {id:"input", label:"任务输入", en:"TASK INPUT"},
  {id:"manage", label:"Manage 规划", en:"TASK PLANNING"},
  {id:"agentic", label:"Agentic 选择", en:"WORKFLOW ROUTING"},
  {id:"components", label:"Component 绑定", en:"SKILL BINDING"},
  {id:"execution", label:"QA 执行", en:"WORKFLOW EXECUTION"},
  {id:"evaluation", label:"效果评估", en:"METRIC EVALUATION"},
  {id:"policy_select", label:"Policy 选择", en:"TARGET SELECTION"},
  {id:"policy_edit", label:"Policy 修改", en:"SKILL REVISION"},
  {id:"validation", label:"执行验证", en:"REVISION VALIDATION"},
  {id:"inheritance", label:"版本继承", en:"NEXT TASK"},
];
const TABS = [{id:"overview",label:"概览"},{id:"inputs",label:"输入"},{id:"prompt",label:"Prompt"},{id:"response",label:"回答"},{id:"logic",label:"执行逻辑"},{id:"evidence",label:"证据"}];
const STATUS_LABELS = {completed:"已完成",success:"已完成",succeeded:"已完成",done:"已完成",passed:"已通过",accepted:"已采用",running:"进行中",in_progress:"进行中",started:"进行中",pending:"待执行",queued:"等待执行",not_started:"待执行",unknown:"未记录",not_recorded:"未记录",failed:"错误",error:"错误",rejected:"已拒绝",skipped:"已跳过",skip:"已跳过",no_change:"无修改",unchanged:"未改动",rolled_back:"已回退",cancelled:"已取消",interrupted:"已中断",partial:"部分完成"};
function statusClass(value) {
  const valueLower = String(value || "unknown").toLowerCase();
  if (["completed","success","succeeded","done","passed","accepted"].includes(valueLower)) return "completed";
  if (["running","in_progress","started"].includes(valueLower)) return "running";
  if (["failed","error"].includes(valueLower)) return "failed";
  if (["rejected","skipped","skip","no_change","unchanged","rolled_back","cancelled","interrupted"].includes(valueLower)) return "skipped";
  return "pending";
}
function statusLabel(value) { return STATUS_LABELS[String(value || "unknown").toLowerCase()] || String(value); }
function badge(value) { return el("span", `status-badge ${statusClass(value)}`, statusLabel(value)); }
function runBadge(value) {
  if(["running","in_progress","started"].includes(String(value || "").toLowerCase())) {
    const node=el("span","status-badge unknown","日志未结束");
    node.title="尚未记录终态；此只读面板不核对后台进程存活，不能据此认定进程仍在运行。";
    return node;
  }
  return badge(value);
}

const initialHash = new URLSearchParams(location.hash.replace(/^#/, ""));
const state = {
  runs:[], run:null, taskId:initialHash.get("task"), nodeId:initialHash.get("node"),
  questionId:initialHash.get("question"), callId:initialHash.get("call"), component:initialHash.get("component"),componentCallId:initialHash.get("component_call"),
  tab:TABS.some(t=>t.id===initialHash.get("tab")) ? initialHash.get("tab") : "overview",
  desiredRun:initialHash.get("run"), detail:null, detailError:null,
  page:1, pageSize:7, questionQuery:"", questionStatus:"all", runQuery:"",
  loading:false, runRequest:0, nodeRequest:0,
};
const currentTask = () => array(state.run?.tasks).find(t=>String(t.id)===String(state.taskId));
const currentQuestion = () => array(currentTask()?.questions).find(q=>String(q.id)===String(state.questionId));
const currentNode = () => FLOW.find(n=>n.id===state.nodeId);
const componentRecordId = (record) => record?.component_call_id || record?.id;
function currentComponentCall(detail=state.detail) {
  const rows=array(detail?.component_calls);
  return rows.find(row=>componentRecordId(row)===state.componentCallId) || detail?.selected_component_call || rows[0] || null;
}
function visibleCalls(detail=state.detail) {
  const calls=array(detail?.calls);
  if(!state.component)return calls;
  const selectedComponentId=componentRecordId(currentComponentCall(detail));
  return calls.filter(call=>{
    if(selectedComponentId)return call.component_call_id===selectedComponentId;
    return [call.component_name,call.component,call.slot].includes(state.component);
  });
}
function visibleSelectedCall(detail=state.detail) {
  const chosen=detail?.selected_call;
  return chosen && visibleCalls(detail).some(call=>call.id===chosen.id) ? chosen : null;
}
const cache = new Map();
const compactMap = matchMedia("(min-width: 1080px) and (max-width: 1699px)");

async function getJSON(path) {
  const response = await fetch(path, {headers:{Accept:"application/json"},cache:"no-store"});
  if (!response.ok) {
    let message = `读取日志失败（HTTP ${response.status}）`;
    try { const body = await response.json(); message = body.error || body.message || message; } catch (_) { /* An HTTP error may have no JSON body. */ }
    throw new Error(message);
  }
  return response.json();
}
function syncHash() {
  const params = new URLSearchParams();
  const values = {run:state.run?.id,task:state.taskId,node:state.nodeId,question:state.questionId,tab:state.tab,call:state.callId,component:state.component,component_call:state.componentCallId};
  for (const [key,value] of Object.entries(values)) if (present(value)) params.set(key,value);
  history.replaceState(null,"",`${location.pathname}${location.search}#${params}`);
}
function notice(message, isError = false) {
  $("global-notice").hidden = !message;
  $("global-notice").textContent = message || "";
  $("global-notice").classList.toggle("is-error",isError);
}
function setLoading(loading) {
  state.loading = loading;
  $("refresh-button").disabled = loading;
  $("refresh-button").querySelector("svg")?.classList.toggle("spin",loading);
  $("source-status").textContent = loading ? "读取中" : "已连接";
}

async function refresh() {
  if (state.loading) return;
  setLoading(true);
  cache.clear();
  notice("");
  try {
    const data = await getJSON("/api/runs");
    state.runs = array(data.runs);
    renderRuns();
    const wantedId = state.run?.id || state.desiredRun;
    const next = state.runs.find(r=>r.id===wantedId) || state.runs[0];
    if (next) await loadRun(next.id, next.id === wantedId);
    else {
      state.run = null; clear($("run-overview"));
      $("workspace").hidden = true; $("empty-state").hidden = false;
      if (array(data.notices).length) notice(data.notices.map(text).join(" · "));
    }
    $("updated-at").textContent = `刷新于 ${new Date().toLocaleTimeString("zh-CN",{hour12:false})}`;
    $("source-status").textContent = "已连接";
  } catch (error) {
    notice(`${error.message}。可点击“刷新日志”重试。`,true);
    $("source-status").textContent = "读取异常";
    if (!state.run) { $("empty-state").hidden = false; $("workspace").hidden = true; }
  } finally {
    setLoading(false);
    if ($("global-notice").classList.contains("is-error")) $("source-status").textContent = "读取异常";
  }
}

async function loadRun(id, preserve = false) {
  const ticket = ++state.runRequest;
  ++state.nodeRequest;
  state.detail = null; state.detailError = null;
  try {
    const run = await getJSON(`/api/runs/${encodeURIComponent(id)}`);
    if (ticket !== state.runRequest) return;
    state.run = run;
    if (!preserve) {
      state.taskId = null; state.nodeId = null; state.questionId = null; state.callId = null; state.component = null;state.componentCallId=null; state.tab = "overview";
      state.page = 1; state.questionQuery = ""; $("question-search").value = "";
    }
    const tasks = array(run.tasks);
    if (!tasks.some(t=>String(t.id)===String(state.taskId))) state.taskId = (tasks.find(t=>statusClass(t.status)==="running") || tasks[0])?.id || null;
    if (!FLOW.some(n=>n.id===state.nodeId)) state.nodeId = null;
    if (state.questionId && !currentQuestion()) { state.questionId = null; state.callId = null; }
    if (state.questionId) {
      const questionIndex = array(currentTask()?.questions).findIndex(q=>String(q.id)===String(state.questionId));
      if (questionIndex>=0) state.page=Math.floor(questionIndex/state.pageSize)+1;
    }
    $("workspace").hidden = !tasks.length;
    $("empty-state").hidden = tasks.length > 0;
    renderRuns(); renderOverview(); renderWorkflow(); renderQuestions(); renderInspector(); syncHash();
    const notes = array(run.notices).map(text);
    notice(notes.length ? notes.join(" · ") : "");
    if (state.nodeId && state.taskId) await loadNode();
  } catch (error) {
    if (ticket === state.runRequest) notice(`${error.message}。可刷新后重新选择运行。`,true);
  }
}

function renderRuns() {
  $("run-count").textContent = number(state.runs.length);
  const query = state.runQuery.toLowerCase();
  const runs = state.runs.filter(r=>`${r.name || ""} ${r.id} ${r.group_id || ""} ${r.batch || ""} ${r.created_at || ""}`.toLowerCase().includes(query));
  const list = $("runs-list");
  clear(list);
  for (const run of runs) {
    const title = el("span","run-card-title",icon("layers"),el("span","run-card-name",run.group_id || run.name || run.id));
    const counts = run.counts || {};
    const item = button(`run-card${state.run?.id===run.id ? " active" : ""}`, [title,el("small","",batchLabel(run)),el("span","run-card-bottom",runBadge(run.status),el("span","",`${number(counts.qa_completed)} QA`))],()=>loadRun(run.id));
    item.setAttribute("aria-current",state.run?.id===run.id ? "true" : "false");
    item.title = `${run.name || run.id} · ${run.group_id || "Group 未记录"} · ${run.batch || "批次未记录"}`;
    list.append(item);
  }
  if (!runs.length) list.append(el("div","side-loading",state.runs.length ? "没有匹配的运行" : "尚未发现实验运行"));
}
function batchLabel(run) {
  const batch=run.batch || "";
  const date=batch.match(/(20\d{2})(\d{2})(\d{2})(?:T(\d{2})(\d{2})(\d{2})Z)?/);
  const supplement=batch.match(/supplement[-_]?([\d]+)/i);
  const kind=supplement ? `补跑 ${supplement[1]}` : batch.includes("two-tasks") ? "双 Task 完整捕获" : batch.includes("ten-groups") ? "多 Group 批次" : batch.replace(/^real[-_]?/, "").replace(/[-_]?20\d{6}.*$/, "") || run.name || run.id;
  if(!date)return kind;
  if(date[4]) {
    const timestamp=new Date(`${date[1]}-${date[2]}-${date[3]}T${date[4]}:${date[5]}:${date[6]}Z`);
    return `${timestamp.toLocaleDateString("zh-CN",{month:"2-digit",day:"2-digit"})} ${timestamp.toLocaleTimeString("zh-CN",{hour:"2-digit",minute:"2-digit",hour12:false})} · ${kind}`;
  }
  return `${date[2]}/${date[3]} · ${kind}`;
}

function statCard(label, value, target, subtitle, iconName, withProgress = false) {
  const valueLine = el("div","stat-number",number(value));
  if (present(target)) valueLine.append(el("small","",`/ ${number(target)}`));
  const node = el("div","stat-card",el("div","stat-label",label,icon(iconName)),valueLine);
  if (withProgress && Number.isFinite(Number(target)) && Number(target)>0 && present(value)) {
    const fill = el("i"); fill.style.width = `${Math.min(100, Math.max(0,Number(value)/Number(target)*100))}%`;
    node.append(el("div","stat-progress",fill));
  } else node.append(el("div","stat-footer",subtitle));
  return node;
}
function renderOverview() {
  const run = state.run; if (!run) return;
  const counts = run.counts || {};
  const nameLine = el("div","run-name-line",el("h2","",run.group_id || "Group 未记录"),runBadge(run.status));
  nameLine.querySelector("h2").title = run.group_id || "Group 未记录";
  const modelLine = (label,value) => el("span","",el("b","model-role",label),present(value) ? value : "此运行未记录");
  const identity = el("div","run-identity",el("div","group-kicker","CURRENT EXPERIMENT"),nameLine,el("div","run-models",modelLine("EXECUTOR",run.executor_model),modelLine("POLICY",run.policy_model)),el("div","run-validation-mode",`验证模式 · ${validationModeLabel(run.revision_validation_mode)}`));
  clear($("run-overview"),identity,statCard("已完成任务",counts.tasks_completed,counts.tasks_target,"任务目标未记录","task",true),statCard("QA 执行",counts.qa_completed,counts.qa_target,"问答目标未记录","questions",true),statCard("逻辑优化轮次",counts.optimization_rounds,null,"按 Task 优化过程统计","evolution"));
  const callCounts=run.call_counts || {};
  const actualCalls=array(run.calls).filter(c=>c.record_kind!=="optimizer_audit");
  const policySelection=actualCalls.filter(c=>c.stage==="policy_selection").length;
  const policyEditing=actualCalls.filter(c=>c.stage==="policy_editing").length;
  const captureCount=actualCalls.filter(c=>c.capture_available).length;
  const micro=(label,value)=>el("span","call-stat",el("span","",label),el("b","",number(value)));
  const strip=el("div","call-stat-strip",micro("Generate",callCounts.generate_calls),micro("Policy 选择 / 修改",null));
  strip.lastChild.querySelector("b").textContent=actualCalls.length ? `${policySelection} / ${policyEditing}` : "未记录";
  strip.append(micro("HTTP 尝试",callCounts.http_attempts),micro("传输重试",callCounts.http_retries),micro("恢复调用",callCounts.recovered_calls),micro("采用修改",counts.revisions_accepted));
  const coverage=el("span","capture-coverage",icon("shield"),`正文捕获 ${number(captureCount)} / ${number(callCounts.generate_calls)}`);
  coverage.title="按 capture_available 统计有正文捕获的 Generate 调用；历史优化审计单独记载。";
  strip.append(coverage);$("run-overview").append(strip);
}

function selectTask(id, nodeId = null) {
  if (String(id) === String(state.taskId) && !nodeId) return;
  state.taskId = id; state.questionId = null; state.callId = null; state.component = null;state.componentCallId=null;
  state.nodeId = nodeId; state.detail = null; state.detailError = null;
  state.tab = nodeId==="inheritance" ? "evidence" : "overview";
  state.page = 1; state.questionQuery = ""; $("question-search").value = "";
  ++state.nodeRequest;
  renderWorkflow();renderQuestions();renderInspector();syncHash();
  if (nodeId) loadNode();
}
function renderWorkflow() {
  const task = currentTask(); if (!task) return;
  $("workflow-task-title").textContent = `Task ${task.index}`;
  clear($("task-status"),badge(task.status));
  clear($("task-tabs"));
  for (const item of array(state.run.tasks)) {
    const tab = button("task-tab",[el("i",`tab-status ${statusClass(item.status)}`),`Task ${item.index}`],()=>selectTask(item.id));
    tab.setAttribute("role","tab");tab.setAttribute("aria-selected",String(item.id===task.id));
    tab.setAttribute("tabindex",item.id===task.id ? "0" : "-1");tab.title = item.id;
    $("task-tabs").append(tab);
  }
  clear($("task-summary"),el("span","",`TASK ID · ${task.id}`),el("span","","Skill ",el("code","",shortHash(task.skill_hash_before))));
  if(task.qa_execution) {
    const scheduling=task.qa_execution;
    const limit=scheduling.final_concurrency_limit ?? scheduling.concurrency_limit;
    $("task-summary").append(el("span","",`${qaModeLabel(scheduling.config?.mode)} · 峰值 ${number(scheduling.peak_in_flight)} · ${scheduling.status==="running" ? "当前" : "最终"}上限 ${number(limit)}`));
  }
  if(task.vector_index)$("task-summary").append(el("span","",`Vector · 建库 ${number(task.vector_index.builds)} · 加载 ${number(task.vector_index.disk_loads)} · 内存命中 ${number(task.vector_index.memory_hits)}`));
  clear($("workflow-map"));
  for (let index=0;index<FLOW.length;index++) {
    const step = FLOW[index];
    const record = array(task.nodes).find(n=>n.id===step.id) || {};
    const label = step.id==="execution" && present(task.question_count) ? `${number(task.question_count)} QA 执行` : step.label;
    const node = button(`flow-node ${statusClass(record.status)}`, [el("span","node-top",el("span","node-icon",icon(step.id)),el("span","node-number",String(index+1).padStart(2,"0"))),el("span","node-title",label),el("span","node-subtitle",step.en),el("span","node-state",statusLabel(record.status))],()=>selectNode(step.id));
    node.setAttribute("aria-pressed",String(state.nodeId===step.id));
    node.setAttribute("aria-label",`${label}，${statusLabel(record.status)}，查看详情`);
    node.title = text(record.summary || label);
    let col = index<5 ? index+1 : 10-index;
    let row=index<5 ? 1 : 2;
    let direction=index===4 ? "down" : index>4 ? "reverse" : "";
    if(compactMap.matches) {
      const positions=[[1,1],[2,1],[3,1],[3,2],[2,2],[1,2],[1,3],[2,3],[3,3],[3,4]];
      [col,row]=positions[index];
      direction=[2,5,8].includes(index) ? "down" : [3,4].includes(index) ? "reverse" : "";
    }
    node.style.gridColumn = String(col);node.style.gridRow = String(row);
    if (index<9) node.append(el("i",`flow-connector${direction ? ` ${direction}` : ""}`));
    $("workflow-map").append(node);
  }
  renderEvolution();
}

function renderEvolution() {
  const tasks = array(state.run?.tasks);clear($("evolution-chain"));
  for (let i=0;i<tasks.length;i++) {
    const task = tasks[i];
    if (i) $("evolution-chain").append(el("span","evolution-arrow",icon("arrow-right")));
    const step = button(`evolution-step${task.id===state.taskId ? " active" : ""}`,[el("span","evolution-step-label",icon("task"),`Task ${task.index}`),el("div","evolution-hash",`输入 ${shortHash(task.skill_hash_before)}`),el("div","evolution-hash",`输出 ${shortHash(task.skill_hash_after)}`)],()=>selectTask(task.id,"inheritance"),`查看 Task ${task.index} 的 Skill 继承证据`);
    $("evolution-chain").append(step);
  }
  const task = currentTask();
  const next = tasks.find(t=>Number(t.index)===Number(task?.index)+1);
  let message = "点击 Task 查看版本哈希、Skill diff、执行验证与下一 Task 的继承关系。";
  if (task?.skill_hash_after && next?.skill_hash_before) message = task.skill_hash_after===next.skill_hash_before ? `Task ${task.index} 的输出版本与 Task ${next.index} 的输入哈希一致，可继续追踪下一 Task 的重新规划。` : `Task ${task.index} 的输出与 Task ${next.index} 的输入哈希不同，可点击查看原始继承证据。`;
  else if (!task?.skill_hash_after) message = `Task ${task?.index || ""} 的输出 Skill 哈希此运行未记录。可查看已记录的修改提案与验证过程。`;
  $("evolution-note").textContent = message;
}

function filteredQuestions() {
  return array(currentTask()?.questions).filter(q=>{
    const matchesText = `${q.id} ${q.question || ""} ${q.prediction || ""}`.toLowerCase().includes(state.questionQuery.toLowerCase());
    return matchesText && (state.questionStatus==="all" || statusClass(q.status)===state.questionStatus);
  });
}
function renderQuestions() {
  const task = currentTask(); if (!task) return;
  const questions = filteredQuestions();
  const totalPages = Math.max(1,Math.ceil(questions.length/state.pageSize));
  state.page=Math.max(1,Math.min(state.page,totalPages));
  $("question-total").textContent = present(task.question_count) ? `${number(task.question_count)} 条` : `${number(array(task.questions).length)} 条已记录`;
  $("qa-completion").textContent = `完成 ${number(task.qa_completed)}${present(task.question_count) ? ` / ${number(task.question_count)}` : ""}`;
  clear($("questions-list"));
  for (const q of questions.slice((state.page-1)*state.pageSize,state.page*state.pageSize)) {
    const metrics = q.metrics && typeof q.metrics==="object" ? Object.entries(q.metrics).filter(([,v])=>typeof v==="number").slice(0,2) : [];
    const metadata = el("span","question-meta",el("code","",q.id),...metrics.map(([k,v])=>el("span","",`${k} ${Number(v).toFixed(3)}`)));
    const row = button(`question-row${String(q.id)===String(state.questionId) ? " active" : ""}`,[el("span","question-index",String(q.index || "·").padStart(2,"0")),el("span","question-content",el("span","question-text",q.question || "问题正文此运行未记录"),metadata),badge(q.status)],()=>selectQuestion(q.id));
    row.setAttribute("aria-pressed",String(String(q.id)===String(state.questionId)));
    $("questions-list").append(row);
  }
  if (!questions.length) $("questions-list").append(el("div","list-empty",array(task.questions).length ? "没有匹配的问答，试试其他关键词。" : "此 Task 尚未记录逐条问答。"));
  const start = questions.length ? (state.page-1)*state.pageSize+1 : 0;
  $("pagination-info").textContent=`显示 ${start}–${Math.min(state.page*state.pageSize,questions.length)} / ${questions.length} 条记录`;
  $("page-info").textContent=`${state.page} / ${totalPages}`;
  $("previous-page").disabled=state.page<=1;$("next-page").disabled=state.page>=totalPages;
}
function focusInspector() {
  if (matchMedia("(max-width: 1079px)").matches) {
    $("inspector").scrollIntoView({behavior:matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth",block:"start"});
    $("inspector").focus({preventScroll:true});
  }
}
function selectNode(id) {
  state.nodeId=id;state.questionId=null;state.callId=null;state.component=null;state.componentCallId=null;
  state.detail=null;state.detailError=null;state.tab="overview";
  renderWorkflow();renderQuestions();renderInspector();syncHash();loadNode();focusInspector();
}
function selectQuestion(id) {
  state.questionId=id;state.nodeId="execution";state.callId=null;state.component=null;state.componentCallId=null;state.tab="overview";
  state.detail=null;state.detailError=null;
  renderWorkflow();renderQuestions();renderInspector();syncHash();loadNode();focusInspector();
}

async function loadNode() {
  const ticket=++state.nodeRequest;
  const params=new URLSearchParams({task_id:state.taskId,node_id:state.nodeId});
  if (state.questionId) params.set("question_id",state.questionId);
  if (state.callId) params.set("call_id",state.callId);
  if (state.component) params.set("component",state.component);
  const path=`/api/runs/${encodeURIComponent(state.run.id)}/node?${params}`;
  $("detail-body").classList.add("loading");
  try {
    const detail=cache.has(path) ? cache.get(path) : await getJSON(path);
    if (ticket!==state.nodeRequest) return;
    cache.set(path,detail);state.detail=detail;state.detailError=null;
    if(state.component)state.componentCallId=componentRecordId(currentComponentCall(detail)) || null;
    state.callId=visibleSelectedCall(detail)?.id || null;
    renderInspector();syncHash();
  } catch(error) {
    if (ticket!==state.nodeRequest) return;
    state.detailError=error.message;state.detail=null;renderInspector();
  } finally { if (ticket===state.nodeRequest) $("detail-body").classList.remove("loading"); }
}

function renderInspector() {
  const detail=state.detail; const task=currentTask();const question=currentQuestion();
  const componentCall=state.component ? currentComponentCall(detail) : null;
  $("inspector-title").textContent=state.component ? state.component : detail?.evidence?.viewing_historical_call ? "历史模型调用" : question ? `QA ${question.index} · 执行轨迹` : detail?.title || currentNode()?.label || "探索这次实验";
  clear($("inspector-status"),state.nodeId ? badge(componentCall?.status || (detail?.evidence?.viewing_historical_call ? detail?.status : question?.status) || detail?.status || array(task?.nodes).find(n=>n.id===state.nodeId)?.status) : el("span","subtle-tag","选择任意环节"));
  $("inspector-context").textContent=state.nodeId ? `Task ${task?.index || "—"}  /  ${state.questionId || state.nodeId}${state.component ? `  /  组件 ${state.component}${componentCall?.slot ? ` · ${componentCall.slot}` : ""}` : detail?.model ? `  /  ${detail.model}` : ""}` : "任务、调用、源码与版本变化，沿着执行路径逐层展开。";
  clear($("inspector-tabs"));
  for (const item of TABS) {
    const tab=button("inspector-tab",state.component && item.id==="response" ? "组件输出" : item.label,()=>{state.tab=item.id;renderInspector();syncHash();});
    tab.setAttribute("role","tab");tab.setAttribute("aria-selected",String(state.tab===item.id));tab.setAttribute("tabindex",state.tab===item.id ? "0" : "-1");
    $("inspector-tabs").append(tab);
  }
  const componentCalls=array(detail?.component_calls);
  $("component-call-selector-wrap").hidden=!state.component || !componentCalls.length;
  clear($("component-call-selector"));clear($("component-call-meta"));
  for(const [index,record] of componentCalls.entries()) {
    const option=el("option","",`${String(index+1).padStart(2,"0")} · ${record.slot || record.component_name || state.component} · ${statusLabel(record.status)} · ${duration(record.seconds ?? record.duration_seconds)}`);
    option.value=componentRecordId(record);option.selected=componentRecordId(record)===componentRecordId(componentCall);$("component-call-selector").append(option);
  }
  if(componentCall)for(const value of [componentRecordId(componentCall),componentCall.scope,duration(componentCall.seconds ?? componentCall.duration_seconds)])if(present(value))$("component-call-meta").append(el("span","",value));
  const calls=visibleCalls(detail);
  $("call-selector-wrap").hidden=!state.nodeId || !calls.length;
  clear($("call-selector"));
  for (const [index,call] of calls.entries()) {
    const option=el("option","",`${String(index+1).padStart(2,"0")} · ${call.role || "角色未记录"} / ${call.stage || "阶段未记录"} / ${call.model || "模型未记录"}`);
    option.value=call.id;option.selected=call.id===(detail?.selected_call?.id || state.callId);$("call-selector").append(option);
  }
  const selected=visibleSelectedCall(detail);
  clear($("call-meta"));
  if(selected) {
    for(const val of [selected.role,selected.model,selected.stage,duration(selected.seconds),present(selected.http_attempt_count) ? `${number(selected.http_attempt_count)} 次 HTTP 尝试` : null]) if(present(val)) $("call-meta").append(el("span","",val));
  }
  clear($("detail-body"));
  if(state.detailError) {
    $("detail-body").append(el("div","detail-notice warning",state.detailError),button("button button-outline",[icon("refresh"),"重试读取"],()=>loadNode()));return;
  }
  if(!state.nodeId) { renderWelcome();return; }
  if(!detail) {
    $("detail-body").append(el("div","missing-record",icon("clock"),"正在读取该环节的执行证据…"));return;
  }
  if(state.component)renderComponentContext(detail);
  switch(state.tab) {
    case "inputs": renderInputs(detail);break;
    case "prompt": renderPrompt(detail);break;
    case "response": renderResponse(detail);break;
    case "logic": renderLogic(detail);break;
    case "evidence": renderEvidence(detail);break;
    default:renderDetailOverview(detail);
  }
}

function section(title,content,iconName=null,copyValue=null) {
  const heading=el("h3","detail-section-title",el("span","",iconName ? icon(iconName) : null,title));
  if(present(copyValue)) heading.append(copyButton(copyValue));
  return el("section","detail-section",heading,content);
}
function copyButton(value) {
  const node=button("copy-button",[icon("copy"),"复制"],async()=>{
    try {
      if(!navigator.clipboard?.writeText) throw new Error("clipboard unavailable");
      await navigator.clipboard.writeText(text(value));
      clear(node,icon("copy"),"已复制");
    } catch (_) { clear(node,icon("info"),"请手动选择复制"); }
  },"复制完整原文");
  return node;
}
function missing(message="此运行未记录") { return el("div","missing-record",icon("info"),message); }
function codeBlock(value,extraClass="") { return nonempty(value) ? el("pre",`detail-code ${extraClass}`.trim(),text(value)) : missing(); }
function capturedBlock(value,extraClass="") {
  if(value===null || value===undefined)return missing();
  return el("pre",`detail-code ${extraClass}`.trim(),value==="" ? "（已记录：空字符串）" : text(value));
}
function validationModeLabel(value) {
  if(!value || value==="unknown")return "此运行未记录";
  return ({execution:"执行验证",execution_only:"执行验证",execution_validation:"执行验证",performance:"性能门槛",performance_gate:"性能门槛",metrics:"指标门槛"})[value] || value;
}
function qaModeLabel(value) {
  return ({serial:"串行",fixed:"固定并行",adaptive:"自适应并行"})[value] || (present(value) ? value : "并行模式未记录");
}
function editModeLabel(value) {
  return ({full:"full · 完整文件",patch:"patch · 精确增量替换"})[value] || "此运行未记录";
}
function warmStartReason(value) {
  const labels={no_committed_history:"没有可继承的已提交 Task",reuse_disabled:"已关闭跨 Task 复用",mode_not_adaptive:"当前为串行或固定并行",history_not_adjacent:"健康记录不属于相邻 Task",previous_task_has_no_warm_start_evidence:"前一 Task 缺少健康并发记录",execution_identity_changed:"模型、路由或资源身份发生变化",previous_task_untrusted_statistics:"前一 Task 的调用统计不可信或不完整",previous_task_missing_evidence:"前一 Task 缺少健康窗口证据",decayed_limit_not_above_configured_initial:"安全衰减后不高于配置起点，使用配置起点",committed_healthy_limit_decayed:"继承已提交 Task 的健康并发，并安全衰减",qa_failure:"主 QA 执行失败",statistics_not_available:"调用统计不可用或覆盖不完整",model_retry_pressure:"出现真实限流或重试压力",insufficient_healthy_windows:"健康饱和窗口不足两个",healthy_saturated_windows:"已有至少两个实测健康饱和窗口"};
  if(labels[value])return labels[value];
  if(typeof value==="string" && value.startsWith("previous_task_"))return `前一 Task：${warmStartReason(value.slice("previous_task_".length))}`;
  return value || "此运行未记录";
}
function renderVectorIndex(summary,preparation,events=[]) {
  if(preparation) {
    const resources=array(preparation.resources);
    const content=el("div","",kv([
      ["状态",statusLabel(preparation.status)],
      ["QA 前准备墙钟时间",present(preparation.duration_seconds) ? duration(preparation.duration_seconds) : null],
      ["准备的共享索引",Array.isArray(preparation.resources) ? resources.length : null],
    ]));
    if(preparation.status==="completed" && resources.length===0)content.append(el("p","section-subline","本 Task 未选择需要提前准备的检索资源。"));
    if(resources.length)content.append(el("details","call-attempt",
      el("summary","","查看共享索引准备证据"),codeBlock(resources)));
    if(preparation.status==="failed")content.append(codeBlock({error_type:preparation.error_type,error:preparation.error}));
    appendSection("检索资源准备",content,"database");
  } else if(!summary)appendSection("检索资源准备",missing("此运行未记录索引提前准备信息。"),"database");
  if(!summary) {
    appendSection("Vector 索引统计",missing(preparation?.status==="completed" && Array.isArray(preparation.resources) && preparation.resources.length===0 ? "本 Task 未记录 vector 检索操作。" : "此范围未记录 vector 建库、加载或查询统计。"),"database");
    return;
  }
  const timings=summary.timings || {};
  const content=el("div","",kv([
    ["建库次数",summary.builds],["磁盘加载次数",summary.disk_loads],["内存命中次数",summary.memory_hits],
    ["缓存命中率（已知来源）",present(summary.cache_hit_rate) ? `${(100*summary.cache_hit_rate).toFixed(1)}%` : null],
    ["来源未记录的操作",summary.cache_source_unknown],["查询次数",summary.queries],
    ["失败操作",summary.failed_operations],
    ["建库累计耗时",present(timings.build_seconds) ? duration(timings.build_seconds) : null],
    ["磁盘加载累计耗时",present(timings.load_seconds) ? duration(timings.load_seconds) : null],
    ["查询累计耗时",present(timings.query_seconds) ? duration(timings.query_seconds) : null],
    ["查询向量编码累计耗时",present(timings.query_embedding_seconds) ? duration(timings.query_embedding_seconds) : null],
    ["相似度搜索累计耗时",present(timings.search_seconds) ? duration(timings.search_seconds) : null],
    ["索引锁等待累计耗时",present(timings.wait_seconds) ? duration(timings.wait_seconds) : null],
  ]),el("p","section-subline","查询耗时包含查询向量编码与相似度搜索，不含索引获取。并行操作的累计耗时会重叠，不能当作 Task 墙钟时间。"));
  if(summary.availability==="partial")content.append(el("p","detail-notice warning","统计字段部分覆盖；未记录的耗时保留未知。次数仅统计已记录的来源。"),codeBlock(summary.timing_coverage));
  if(array(events).length) {
    const details=el("details","call-attempt",el("summary","",`${events.length} 条索引操作证据`));
    for(const operation of events) {
      const source=({built:"首次建库",disk:"磁盘命中",memory:"内存命中"})[operation.cache_source] || "来源未记录";
      details.append(el("details","call-attempt",el("summary","",`${operation.operation==="prepare" ? "准备" : "查询"} · ${source} · ${statusLabel(operation.status)} · ${duration(operation.duration_seconds)}`),codeBlock(operation)));
    }
    content.append(details);
  }
  appendSection("Vector 索引统计",content,"database");
}
function renderQAScheduling(scheduling) {
  if(!scheduling) {
    appendSection("QA 调度",missing("此运行未记录 QA 并发统计。"),"execution");return;
  }
  const config=scheduling.config || {};
  const limit=scheduling.final_concurrency_limit ?? scheduling.concurrency_limit;
  const warm=scheduling.warm_start;
  const evidence=scheduling.warm_start_evidence;
  const content=el("div","",kv([
    ["模式",qaModeLabel(config.mode)],
    ["配置起始并发",scheduling.configured_initial_concurrency ?? config.initial_workers],
    ["本 Task 实际起点",scheduling.effective_initial_concurrency],
    ["实际在途峰值",scheduling.peak_in_flight],
    [scheduling.status==="running" ? "当前并发上限" : "最终并发上限",limit],
    ["配置并发范围",present(config.min_workers) && present(config.max_workers) ? `${config.min_workers}–${config.max_workers}` : null],
    ["完成 QA",scheduling.completed_examples],
    ["QA 执行墙钟时间",present(scheduling.wall_seconds) ? duration(scheduling.wall_seconds) : null],
  ]),el("p","section-subline","并发峰值表示同时在途的题数。"));
  if(warm) {
    content.append(kv([
      ["跨 Task 起点复用",warm.enabled===false ? "已关闭" : warm.reused===true ? "已复用" : warm.reused===false ? "冷启动" : null],
      ["起点依据",warmStartReason(warm.reason)],
      ["来源 Task",warm.source_task_id],
      ["来源健康并发",warm.healthy_limit],
      ["安全衰减级数",warm.decay_steps],
      ["衰减后起点",warm.starting_limit],
    ]));
    content.append(el("p","section-subline","复用已确认的健康并发并保守衰减；新 Task 重新测量吞吐基准，不继承尾部峰值或未确认的探测上限。"));
  } else content.append(el("p","section-subline","此运行未记录跨 Task 起点继承证据。"));
  if(evidence)content.append(el("details","call-attempt",el("summary","","下一 Task 可复用的健康并发证据"),kv([
    ["可供复用",evidence.eligible===true ? "是" : evidence.eligible===false ? "否" : null],
    ["依据",warmStartReason(evidence.reason)],
    ["确认健康并发",evidence.healthy_limit],
    ["完整健康饱和窗口",evidence.confirmed_saturated_windows],
  ]),codeBlock(evidence)));
  const reasons={probe_capacity:"探测更多并发容量",model_retry_pressure:"模型请求出现重试，降低并发",latency_inflation:"单题延迟升高，降低并发",probe_no_throughput_gain:"增加并发未提升吞吐，退回原上限",throughput_improved:"吞吐提升",sustained_throughput_drop:"连续两窗吞吐下降，降低并发",throughput_drop_unconfirmed:"单窗吞吐下降，保持并发继续观察",throughput_plateau:"吞吐持平，保持并发",cooldown:"冷却观察",at_configured_ceiling:"达到配置上限",hold:"保持并发"};
  const changes=array(scheduling.adjustments);
  if(changes.length) {
    const details=el("details","call-attempt",el("summary","",`${changes.length} 条调度调整记录`));
    for(const change of changes)details.append(el("p","detail-description",`完成 ${number(change.completed_examples)} 题 · ${number(change.previous_limit)} → ${number(change.concurrency_limit)} · ${reasons[change.reason] || change.reason || "原因未记录"}`));
    content.append(details);
  }
  appendSection("QA 调度",content,"execution");
}
function kv(rows) {
  const list=el("dl","detail-kv");
  for(const [label,value] of rows) list.append(el("div","",el("dt","",label),el("dd","",text(value))));
  return list;
}
function appendSection(title,content,iconName=null,copyValue=null) { $("detail-body").append(section(title,content,iconName,copyValue)); }
function appendNotices(notices) {
  for(const message of array(notices)) $("detail-body").append(el("div","detail-notice",text(message)));
}
function returnToQuestion() {
  state.component=null;state.componentCallId=null;state.callId=null;state.detail=null;
  renderInspector();syncHash();loadNode();
}
function renderComponentContext(detail) {
  const controls=el("div","source-controls",el("span","",`组件 · ${state.component}`),button("button button-outline small","返回整个 QA",returnToQuestion));
  $("detail-body").append(controls);
  if(!array(detail.component_calls).length && !visibleCalls(detail).length)$("detail-body").append(el("div","detail-notice warning","QA级调用，未逐组件关联。此运行未记录该组件的独立函数输入输出或专属 Prompt；返回整个 QA 可手动查看 QA 级调用。"));
}
function renderComponentRecording(record) {
  if(!record)return;
  if(nonempty(record.omissions) || nonempty(record.inputs_recording) || nonempty(record.output_recording))appendSection("记录范围与省略",codeBlock({inputs_recording:record.inputs_recording,output_recording:record.output_recording,omissions:record.omissions,limits:record.limits}),"info");
}
function renderWelcome() {
  const task=currentTask();
  $("detail-body").append(el("div","inspector-intro",el("span","intro-icon",icon("cursor")),el("div","",el("h3","","跟随执行路径，查看完整证据"),el("p","","点击左侧任意环节，查看其输入、模型 Prompt、回答和真实执行逻辑。每条 QA 都可以独立追踪。"))));
  appendSection("当前任务",kv([["Task",task?.id],["执行状态",statusLabel(task?.status)],["已完成 QA",`${number(task?.qa_completed)}${present(task?.question_count) ? ` / ${number(task.question_count)}` : ""}`],["输入版本",task?.skill_hash_before],["验证模式",validationModeLabel(state.run?.revision_validation_mode)],["实验版本",state.run?.experiment_version && state.run.experiment_version!=="unknown" ? state.run.experiment_version : null]]),"task");
  renderQAScheduling(task?.qa_execution);
  renderVectorIndex(task?.vector_index,task?.retrieval_resources);
  appendSection("可探索的执行证据",el("div","",el("p","detail-description","Manage 与 Agentic 决策 → Component 绑定 → 逐题执行与评估 → Policy 修改 → 验证与跨任务继承。"),el("p","section-subline","历史运行未捕获的 Prompt 或模型回答会显示“此运行未记录”。")),"layers");
  const calls=array(state.run?.calls).filter(c=>c.task_id===task?.id);
  appendSection("调用记录",kv([["当前 Task",`${calls.length} 条已记录调用`],["查看方法","点击流程环节，再从调用选择器选择完整记录"]]),"code");
}

function renderDetailOverview(detail) {
  const question=currentQuestion();
  if(state.component) {
    const record=currentComponentCall(detail);
    appendSection("组件函数调用",kv([["组件",state.component],["函数调用 ID",componentRecordId(record)],["插槽",record?.slot],["状态",record ? statusLabel(record.status) : null],["耗时",record ? duration(record.seconds ?? record.duration_seconds) : null]]),"components");
    appendSection("组件实际输入",record ? capturedBlock(record.inputs) : missing("此运行未记录逐组件函数输入。"),"input",record?.inputs);
    appendSection("组件实际输出",record ? capturedBlock(record.output,"response-code") : missing("此运行未记录逐组件函数输出。"),"execution",record?.output);
    renderComponentRecording(record);
    if(detail.vector_index)renderVectorIndex(detail.vector_index,null,detail.vector_index_events);
    renderCallList(visibleCalls(detail));appendNotices(record?.notices);appendNotices(detail.notices);return;
  }
  if(detail.description) $("detail-body").append(el("div","inspector-intro",el("span","intro-icon",icon(state.nodeId)),el("div","",el("h3","",currentNode()?.label || detail.title),el("p","",detail.description))));
  if(state.nodeId==="execution" && !detail.evidence?.viewing_historical_call)renderQAScheduling(detail.qa_execution);
  if(["components","execution"].includes(state.nodeId) && !detail.evidence?.viewing_historical_call)renderVectorIndex(detail.vector_index,detail.retrieval_resources,detail.vector_index_events);
  if(question && !detail.evidence?.viewing_historical_call) {
    appendSection("问题",el("p","question-hero",question.question || "此运行未记录"),"questions",question.question);
    appendSection("模型回答",codeBlock(question.prediction,"response-code"),null,question.prediction);
    if(nonempty(question.gold_answers)) appendSection("仅供评估的标注",codeBlock(question.gold_answers));
    renderMetrics(question.metrics);
    renderTimings(question.component_timings);
  } else {
    appendSection("环节信息",kv([["状态",statusLabel(detail.status)],["模型",detail.model],["调用数量",array(detail.calls).length],["任务 ID",detail.task_id || state.taskId]]),"info");
    appendSection("实际输出",codeBlock(detail.output),null,nonempty(detail.output) ? detail.output : null);
    if(["policy_edit","validation","inheritance"].includes(state.nodeId)) renderVersions(detail.skill_versions,state.nodeId==="policy_edit");
    if(state.nodeId==="validation") appendSection("执行验证",codeBlock(detail.verification),"validation");
  }
  renderCallList(detail.calls);
  appendNotices(detail.notices);
}
function renderInputs(detail) {
  if(state.component) {
    const record=currentComponentCall(detail);
    appendSection("组件实际函数输入",record ? capturedBlock(record.inputs) : missing("此运行未记录逐组件函数输入；QA 级输入不能当作此组件的实际输入。"),"input",record?.inputs);
    if(record)appendSection("输入正文校验",kv([["SHA256",record.inputs_sha256 || record.input_sha256],["校验",hashMatchLabel(record.inputs_hash_matches)]]),"shield");
    if(record?.source)appendSection("执行源码证据",kv([["路径",record.source.path],["SHA256",record.source.sha256],["校验",hashMatchLabel(record.source.hash_matches)]]),"code");
    renderComponentRecording(record);
    appendNotices(record?.notices);appendNotices(detail.notices);return;
  }
  if(detail.description) $("detail-body").append(el("p","section-subline",detail.description));
  appendSection("环节输入",codeBlock(detail.inputs),"input",nonempty(detail.inputs) ? detail.inputs : null);
  if(currentQuestion() && !detail.evidence?.viewing_historical_call) {
    const q=currentQuestion();
    appendSection("逐题记录",kv([["Question ID",q.id],["问题",q.question],["选定组件",q.selection]]));
    if(nonempty(q.gold_answers))appendSection("仅供评估的标注",codeBlock(q.gold_answers));
  }
  if(nonempty(detail.evaluation_only_annotations))appendSection("仅供评估的标注",codeBlock(detail.evaluation_only_annotations));
  appendSection("输入 Skill 版本",codeBlock(currentTask()?.skill_hash_before));
}
function renderPrompt(detail) {
  const call=visibleSelectedCall(detail);
  if(!call) {
    $("detail-body").append(missing(state.component ? "没有可可靠归属于该组件函数调用的 LLM Prompt。组件实际函数输入可在“输入”中查看。" : "此运行未记录该环节的模型调用。部分本地组件可直接执行，不产生模型 Prompt。"));
    appendNotices(detail.notices);return;
  }
  appendSection("System Prompt",capturedBlock(call.system_prompt,"prompt-code"),"code",call.system_prompt);
  appendSection("User Prompt · 完整原文",capturedBlock(call.prompt,"prompt-code"),"questions",call.prompt);
  appendSection("Prompt 正文校验",kv([["System SHA256",call.system_prompt_sha256],["System 校验",hashMatchLabel(call.system_prompt_hash_matches)],["User SHA256",call.prompt_sha256],["User 校验",hashMatchLabel(call.prompt_hash_matches)]]),"shield");
  if(nonempty(call.messages)) appendSection("原始消息",codeBlock(call.messages));
  appendSection("调用标识",kv([["Call ID",call.id],["角色",call.role],["模型",call.model],["阶段 / Scope",[call.stage,call.scope].filter(present).join(" / ") || null],["Skill 哈希",call.skill_hash]]));
  appendNotices(call.notices);appendNotices(detail.notices);
}
function renderResponse(detail) {
  if(state.component) {
    const record=currentComponentCall(detail);
    appendSection("组件实际函数输出",record ? capturedBlock(record.output,"response-code") : missing("此运行未记录逐组件函数输出；QA 最终预测不能当作此组件的输出。"),"execution",record?.output);
    if(record) {
      appendSection("函数调用证据",kv([["函数调用 ID",componentRecordId(record)],["状态",statusLabel(record.status)],["耗时",duration(record.seconds ?? record.duration_seconds)],["输出 SHA256",record.output_sha256],["输出校验",hashMatchLabel(record.output_hash_matches)],["源码 SHA256",record.source?.sha256]]),"code");
      renderComponentRecording(record);
      appendNotices(record.notices);
    }
  }
  const call=visibleSelectedCall(detail);
  if(!call) {
    if(!state.component)$("detail-body").append(missing("此运行未记录可归属到该环节的模型回答。"));
    appendNotices(detail.notices);return;
  }
  appendSection("模型回答 · 完整原文",capturedBlock(call.response,"response-code"),"questions",call.response);
  appendSection("回答正文校验",kv([["SHA256",call.response_sha256],["校验",hashMatchLabel(call.response_hash_matches)]]),"shield");
  appendSection("调用结果",kv([["状态",statusLabel(call.status)],["模型",call.model],["总耗时",duration(call.seconds)],["HTTP 尝试次数",call.http_attempt_count],["捕获时间",call.timestamp]]),"clock");
  const attempts=array(call.http_attempts);
  if(attempts.length) {
    const holder=el("div");
    attempts.forEach((attempt,index)=>{
      const block=el("details","call-attempt");
      block.append(el("summary","",`传输事件 ${index+1}${attempt.event ? ` · ${attempt.event}` : ""}${attempt.status ? ` · ${attempt.status}` : ""}`),codeBlock(attempt));holder.append(block);
    });appendSection("HTTP 尝试与重试证据",holder,"evolution");
  } else appendSection("HTTP 尝试与重试证据",missing());
  appendNotices(call.notices);appendNotices(detail.notices);
}
function hashMatchLabel(value) { return value===true ? "与捕获哈希一致" : value===false ? "哈希不一致 · 请核对原始证据" : "此运行未记录校验依据"; }

function renderMetrics(metrics) {
  if(!nonempty(metrics)) { appendSection("评估指标",missing());return; }
  let shown=false;
  const chips=(values)=>el("div","metric-chips",Object.entries(values).filter(([,val])=>typeof val==="number").map(([key,val])=>el("div","metric-chip",el("span","",key),el("strong","",Number(val).toFixed(4)))));
  for(const [key,label] of [["retrieval","检索指标"],["generation","生成指标"]]) {
    if(nonempty(metrics[key]) && Object.values(metrics[key]).some(val=>typeof val==="number")) {
      appendSection(label,chips(metrics[key]),"evaluation");shown=true;
    }
  }
  if(Object.values(metrics).some(val=>typeof val==="number")) {appendSection("评估指标",chips(metrics),"evaluation");shown=true;}
  if(!shown)appendSection("指标原始记录",codeBlock(metrics));
}
function timingEntries(timings) {
  if(!nonempty(timings))return [];
  if(Array.isArray(timings))return timings.map((row,index)=>({name:row.component || row.skill || row.name || row.component_id || `component ${index+1}`,seconds:row.seconds ?? row.duration_seconds ?? row.duration ?? row.elapsed_seconds,raw:row}));
  return Object.entries(timings).map(([name,row])=>({name,seconds:typeof row==="number" ? row : row?.seconds ?? row?.duration_seconds ?? row?.duration ?? row?.elapsed_seconds,raw:row}));
}
function renderTimings(timings) {
  const entries=timingEntries(timings);
  if(!entries.length)return;
  const max=Math.max(...entries.map(t=>Number(t.seconds)||0),.001);
  const holder=el("div");
  for(const item of entries) {
    const fill=el("i");fill.style.width=`${Math.max(0,(Number(item.seconds)||0)/max*100)}%`;
    const row=button("timing-row timing-button",[el("span","",item.name),el("span","timing-bar",fill),el("span","timing-value",present(item.seconds) ? duration(item.seconds) : "未记录")],()=>{
      state.component=item.name;state.componentCallId=item.raw?.component_call_id || null;state.callId=null;state.tab="overview";state.detail=null;renderInspector();syncHash();loadNode();
    },`查看组件 ${item.name} 的实际输入、输出及执行源码`);
    row.title=text(item.raw);holder.append(row);
  }
  appendSection("组件耗时",holder,"clock");
}
function renderCallList(calls) {
  if(!array(calls).length)return;
  const list=el("div");
  for(const call of calls) {
    const node=button("call-list-button",[el("span","",`${call.role || "角色未记录"} · ${call.model || "模型未记录"}`,el("small","",`${call.stage || "阶段未记录"}  /  ${duration(call.seconds)}  /  ${statusLabel(call.status)}`)),icon("chevron-right")],()=>{state.callId=call.id;state.tab="prompt";state.detail=null;renderInspector();syncHash();loadNode();});
    list.append(node);
  }
  appendSection(`模型调用 · ${calls.length}`,list,"code");
}

function renderLogic(detail) {
  const question=currentQuestion();
  if(question && !state.component && !detail.evidence?.viewing_historical_call) {
    appendSection("逐题执行 Trace",codeBlock(question.trace),"layers",nonempty(question.trace) ? question.trace : null);
    renderTimings(question.component_timings);
    if(nonempty(question.component_timings))appendSection("组件执行原始记录",codeBlock(question.component_timings));
  }
  const sources=array(detail.logic_sources);
  if(sources.length) {
    const holder=el("div");
    for(const [index,source] of sources.entries()) {
      const block=el("details","logic-source");
      block.open=index===0;
      const hashNote=source.hash_matches===false ? " · 与执行捕获哈希不一致" : source.hash_matches===true ? " · 执行哈希匹配" : "";
      const provenance=({captured_execution:"执行时捕获 · 哈希匹配",frozen_snapshot:"运行快照",current_reference:"当前参考源码 · 非历史快照"})[source.source_kind] || source.provenance || "来源类型未记录";
      const summary=el("summary","",source.label || "执行源码",el("span","",provenance),el("span","",source.path || "路径未记录"),el("span","",`SHA256 ${source.sha256 || "未记录"}${hashNote}`));
      block.append(summary,codeBlock(source.text));
      if(source.hash_matches===false) block.append(el("div","detail-notice warning","以下为当前源码，其哈希与运行时捕获不一致。请结合原始证据判断。"));
      holder.append(block);
    }
    appendSection("Skill 与执行源码",holder,"code");
  } else appendSection("Skill 与执行源码",missing());
  if(state.component) appendSection("选定组件",codeBlock(state.component));
  if(nonempty(question?.selection))appendSection("组件绑定记录",codeBlock(question.selection));
  appendNotices(detail.notices);
}

function renderVersions(versions,includeDiff=true) {
  if(!versions)return;
  const hashFlow=el("div","hash-flow",el("code","",shortHash(versions.before)),icon("arrow-right"),el("code","",shortHash(versions.after)));
  appendSection("Skill 版本",hashFlow,"evolution");
  const sourceLabels={selected_model_call:"所选模型调用",optimization_report:"优化报告",revision_report:"修订报告",optimization_audit:"优化审计",revision_proposal:"修订提案",historical_model_call:"历史模型调用"};
  appendSection("Policy 编辑协议",kv([["实际模式",editModeLabel(versions.edit_mode)],["模式证据来源",sourceLabels[versions.edit_mode_source] || null]]),"policy_edit");
  if(versions.edit_mode==="patch")appendSection("原始 Patch 动作",nonempty(versions.raw_patch) ? codeBlock(versions.raw_patch) : missing("此范围未记录可归属的原始 Patch 动作；可在回答页查看模型原文。"),"code",versions.raw_patch);
  if(nonempty(versions.revision))appendSection("修改记录",codeBlock(versions.revision));
  if(includeDiff) {
    if(nonempty(versions.diff)) {
      const pre=el("pre","detail-code");
      for(const line of text(versions.diff).split("\n"))pre.append(el("span",line.startsWith("+")&&!line.startsWith("+++") ? "diff-add" : line.startsWith("-")&&!line.startsWith("---") ? "diff-remove" : "",`${line}\n`));
      appendSection(versions.edit_mode==="patch" ? "Patch 物化后的 Skill Diff" : "Skill Diff",pre,"policy_edit",versions.diff);
    } else appendSection("Skill Diff",missing());
    if(nonempty(versions.proposal))appendSection(versions.edit_mode==="patch" ? "物化文件提案与补丁来源" : "修改提案",codeBlock(versions.proposal));
  }
}
function renderEvidence(detail) {
  appendNotices(detail.notices);
  if(detail.selected_call)appendNotices(detail.selected_call.notices);
  appendSection("记录来源",codeBlock(detail.evidence),"database");
  if(state.component && currentComponentCall(detail))appendSection("组件调用原始证据",codeBlock(currentComponentCall(detail)),"components");
  appendSection("运行约定",kv([["验证模式",validationModeLabel(state.run?.revision_validation_mode)],["实验版本",state.run?.experiment_version && state.run.experiment_version!=="unknown" ? state.run.experiment_version : null]]));
  renderVersions(detail.skill_versions,true);
  if(["policy_edit","validation","inheritance"].includes(state.nodeId))appendSection("执行验证记录",codeBlock(detail.verification),"validation");
  if(state.nodeId==="inheritance")appendSection("下一 Task 继承证据",codeBlock(detail.output),"inheritance");
  const events=array(detail.events);
  if(events.length) {
    const holder=el("ol","event-list");
    for(const event of events) {
      const item=el("li","event-item",el("span","event-time",event.timestamp || event.time || "时间未记录"),el("span","",event.event || event.type || "事件"));
      const expandable=el("details","call-attempt",el("summary","","查看事件原文"),codeBlock(event));
      item.append(expandable);holder.append(item);
    }
    appendSection(`任务事件 · ${events.length}`,holder,"clock");
  } else appendSection("任务事件",missing());
}

function handleTabKeys(event) {
  if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;
  const tabs=Array.from(event.currentTarget.querySelectorAll("[role=tab]"));
  const current=tabs.indexOf(document.activeElement);if(current<0)return;
  event.preventDefault();
  const next=event.key==="Home" ? 0 : event.key==="End" ? tabs.length-1 : (current+(event.key==="ArrowRight" ? 1 : -1)+tabs.length)%tabs.length;
  tabs[next].focus();tabs[next].click();
  const newTabs=Array.from(event.currentTarget.querySelectorAll("[role=tab]"));newTabs[next]?.focus();
}

document.querySelectorAll("[data-icon]").forEach(node=>node.replaceWith(icon(node.dataset.icon)));
$("refresh-button").addEventListener("click",refresh);
$("empty-refresh").addEventListener("click",refresh);
$("run-search").addEventListener("input",event=>{state.runQuery=event.target.value;renderRuns();});
$("question-search").addEventListener("input",event=>{state.questionQuery=event.target.value;state.page=1;renderQuestions();});
$("question-status-filter").addEventListener("change",event=>{state.questionStatus=event.target.value;state.page=1;renderQuestions();});
$("previous-page").addEventListener("click",()=>{state.page--;renderQuestions();});
$("next-page").addEventListener("click",()=>{state.page++;renderQuestions();});
$("call-selector").addEventListener("change",event=>{state.callId=event.target.value;state.detail=null;renderInspector();syncHash();loadNode();});
$("component-call-selector").addEventListener("change",event=>{
  state.componentCallId=event.target.value;
  const matchedCall=visibleCalls()[0];
  state.callId=matchedCall?.id || null;
  if(matchedCall && matchedCall.id!==state.detail?.selected_call?.id) {
    state.detail=null;renderInspector();syncHash();loadNode();
  } else {renderInspector();syncHash();}
});
$("task-tabs").addEventListener("keydown",handleTabKeys);
$("inspector-tabs").addEventListener("keydown",handleTabKeys);
compactMap.addEventListener("change",()=>{if(state.run)renderWorkflow();});
window.addEventListener("hashchange",()=>{
  const values=new URLSearchParams(location.hash.replace(/^#/,""));
  if(values.get("run") && state.runs.some(r=>r.id===values.get("run"))) {
    state.taskId=values.get("task");state.nodeId=values.get("node");state.questionId=values.get("question");state.callId=values.get("call");state.component=values.get("component");state.componentCallId=values.get("component_call");
    state.tab=TABS.some(t=>t.id===values.get("tab")) ? values.get("tab") : "overview";
    loadRun(values.get("run"),true);
  }
});
refresh();
