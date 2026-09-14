'use strict';
let state = {}, token = '', selectedJob = null, selectedComparison = null, inspectedAnswer = null;
const el = id => document.getElementById(id);
const pretty = value => JSON.stringify(value, null, 2);
const text = (id, value) => { el(id).textContent = typeof value === 'string' ? value : pretty(value); };
async function api(path, body, retrySession = true) {
  const response = await fetch('/api/' + path, {method: body === undefined ? 'GET' : 'POST', headers: {'Content-Type':'application/json', 'X-Workbench-Token':token}, ...(body === undefined ? {} : {body:JSON.stringify(body)})});
  const data = await response.json();
  if (response.status === 403 && data.detail === 'Open the local dashboard first' && path !== 'session' && retrySession) {
    token = (await api('session', undefined, false)).token;
    return api(path, body, false);
  }
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : pretty(data.detail)); return data;
}
async function upload(path, file, retrySession = true) {
  const response = await fetch('/api/' + path, {method:'POST', headers:{'Content-Type':'application/octet-stream','X-Workbench-Token':token}, body:file});
  const data = await response.json();
  if (response.status === 403 && data.detail === 'Open the local dashboard first' && retrySession) { token = (await api('session', undefined, false)).token; return upload(path, file, false); }
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : pretty(data.detail)); return data;
}
async function action(fn) { try { text('status', 'Working…'); await fn(); text('status', 'Ready — local workspace. No training starts without Start job.'); } catch (e) { text('status', e.message); } }
function options(id, values, empty, preferred) {
  const select=el(id), prior=select.value; select.replaceChildren();
  if(empty !== undefined) select.add(new Option(empty,''));
  values.forEach(v=>select.add(new Option(v.label,v.id)));
  if([...select.options].some(o=>o.value===prior)) select.value=prior;
  else if([...select.options].some(o=>o.value===preferred)) select.value=preferred;
}
function detail(parent, title, value) {
  const d=document.createElement('details'), s=document.createElement('summary'), p=document.createElement('pre');
  s.textContent=title; p.textContent=pretty(value);d.append(s,p);parent.append(d);
}
const minReportable = () => state.evaluation_policy?.min_reportable_slice ?? 10;
const lowerIsBetter = name => (state.evaluation_policy?.lower_is_better || ['confidence_brier_score']).includes(name);
const ELIGIBILITY = {submitted:'Comment saved. Add a schema-valid correction with expected checks to create a development check.',invalid_correction:'The correction failed the schema, expected checks or training-admission rules. See diagnostics.',protected_split_or_group:'This answer comes from a validation, test or OOT case or group. It can become a development check but never training data. Evaluate the Train split to capture training feedback.',needs_expected_results:'Add expected checks so the correction can be tested.',untriaged:'Choose a cause. Feedback with an unknown cause never enters training.',component_feedback:'Only model_behaviour corrections enter training. Fix this in the named component instead.',eligible_for_batch:'Eligible for the next feedback fragment.'};
function duration(seconds) {
  if(!Number.isFinite(seconds))return '';const t=Math.max(0,Math.round(seconds)),h=Math.floor(t/3600),m=Math.floor(t%3600/60),sec=t%60;
  return h?h+'h '+m+'m':m?m+'m '+sec+'s':sec+'s';
}
function jobDuration(j) {
  const start=Date.parse(j.started_at);if(!Number.isFinite(start))return j.status==='queued'?'waiting':'';
  const end=j.finished_at?Date.parse(j.finished_at):Date.now();return (j.finished_at?'took ':'running for ')+duration((end-start)/1000);
}
function timingText(t) {
  if(!t)return '';const parts=[];
  if(t.finished_at&&Number.isFinite(t.elapsed_seconds))parts.push('took '+duration(t.elapsed_seconds));
  else if(Number.isFinite(t.elapsed_seconds))parts.push('elapsed '+duration(t.elapsed_seconds));
  if(Number.isFinite(t.seconds_per_unit))parts.push(duration(t.seconds_per_unit)+' per '+t.unit);
  if(Number.isFinite(t.eta_seconds))parts.push('about '+duration(t.eta_seconds)+' left');
  return parts.length?' · '+parts.join(' · '):'';
}
function versionLabel(v) {return v.name+(state.active?.[v.task]===v.id?' · active':'')+(state.recommended?.[v.task]===v.id?' · recommended':'');}
function jobRole(j) {return j.spec.comparison_role?j.spec.comparison_role+' evaluate':j.spec.kind;}
function answerLabel(a) {const source=a.identity?.source||'',model=a.identity?.adapter_path?'adapter':'base model',role=a.comparison_role||(a.job_kind==='regression'?'regression':a.job_kind==='ask_plan'?'ask plan draft':a.memory_status==='verified'||source==='human correction'?'ask verified':a.job_kind==='ask'?'ask · '+model:a.identity?.adapter_path?'adapter':source?'imported':'base model');return role+' · '+(a.job_id?a.job_id.slice(0,8)+' · ':'')+a.case.split+' · '+a.case.case_id+' · '+a.case.question.slice(0,80);}
function selections() {
  const task=el('task').value;
  options('datasetSelect',(state.datasets||[]).filter(d=>d.manifest.task===task).map(d=>({id:d.id,label:d.manifest.name||d.id.slice(0,12)})),'Select registered data');
  options('model',(state.models||[]));
  options('version',(state.versions||[]).filter(v=>v.task===task&&v.schema.additionalProperties===false).map(v=>({id:v.id,label:versionLabel(v)})),undefined,state.active[task]);
  const recommended=state.recommended?.[task],recommendedVersion=(state.versions||[]).find(v=>v.id===recommended);
  text('versionNotice',recommended&&el('version').value!==recommended?'The selected version is not the current recommended contract ('+(recommendedVersion?.name||recommended.slice(0,8))+'). Runs freeze the version selected here.':'');
  options('adapter',(state.jobs||[]).filter(j=>j.status==='completed'&&j.spec.kind==='train'&&j.spec.task===task).map(j=>({id:j.id,label:j.id.slice(0,12)})),'Base model');
}
async function refresh() {
  state=await api('state');selections();askSelections();
  options('answerSelect',state.answers.slice().reverse().map(a=>({id:a.id,label:answerLabel(a)})),'Select an answer');
  if(inspectedAnswer&&el('answerSelect').value!==inspectedAnswer)inspectAnswer();
  options('editVersion',state.versions.map(v=>({id:v.id,label:v.task+' / '+versionLabel(v)})));
  const evaluations=state.jobs.filter(j=>j.status==='completed'&&['evaluate','regression'].includes(j.spec.kind)).map(j=>({id:j.id,label:(j.spec.comparison_role||j.spec.kind)+' / '+j.spec.task+' / '+j.id.slice(0,12)}));
  ['evalRun','compareBase','compareCandidate'].forEach(id=>options(id,evaluations,'Select evaluation'));
  renderCandidates();renderComparisons();renderReplays();
  options('buildBase',(state.datasets||[]).filter(d=>d.manifest.format==='credit-workbench-v2').map(d=>({id:d.id,label:(d.manifest.name||d.id.slice(0,12))+' · '+d.manifest.task})),'Select a V2 dataset');
  const list=el('datasetList');list.replaceChildren();
  if(!state.datasets.length){list.className='empty';list.textContent='No datasets registered. Phase-2 data can be added later.';}else{list.className='';state.datasets.forEach(d=>detail(list,d.manifest.name||d.id.slice(0,12),{task:d.manifest.task,contract_version:d.contract_version,contract_warnings:d.contract_warnings,counts:d.counts,taxonomy_counts:d.taxonomy_counts,diversity:d.diversity,missing_targets:d.missing_targets,missing_expectations:d.missing_expectations,token_lengths:tokenSummary(d),hash:d.hash}));}
  const jobs=el('jobList');jobs.replaceChildren();
  if(!state.jobs.length)jobs.textContent='No runs. Training and evaluation are Not evaluated.';
  state.jobs.slice().reverse().forEach(j=>{const row=document.createElement('div');row.className='row';const b=document.createElement('button');b.textContent=j.spec.task+' · '+jobRole(j)+' · '+j.status+' · '+j.id.slice(0,8)+(jobDuration(j)?' · '+jobDuration(j):'');b.onclick=()=>action(()=>showJob(j.id));row.append(b);if(['queued','running','stopping'].includes(j.status)){const stop=document.createElement('button');stop.textContent='Stop';stop.onclick=()=>action(async()=>{await api('jobs/'+j.id+'/stop',{});await refresh();});row.append(stop);}jobs.append(row);});
  const history=el('feedbackList');history.replaceChildren();history.className=state.feedback.length?'':'empty';
  if(!state.feedback.length)history.textContent='No feedback submitted.';
  state.feedback.slice().reverse().forEach(f=>detail(history,f.cause+' · '+(f.eligibility_status||f.regression_status)+' · '+f.created_at,{explanation:ELIGIBILITY[f.eligibility_status]||null,answer:answerLabel(f.snapshot),comment:f.comment,diagnostics:f.diagnostics,has_independent_expectations:f.has_independent_expectations,eligible_for_training:f.eligible_for_training,protected_split_or_group:f.protected_split_or_group,recommendations:f.recommendations}));
}
function tokenSummary(d) {
  const p=(state.preflights||[]).filter(x=>x.dataset_id===d.id).at(-1);if(!p)return d.token_lengths||'Run preflight to measure';
  const v=(state.versions||[]).find(x=>x.id===p.version_id);
  return {max_tokens:p.max_tokens,p95_tokens:p.p95_tokens,min_assistant_tokens:p.min_assistant_tokens,examples:p.examples,max_seq_length:p.max_seq_length,version:v?.name||p.version_id,measured_at:p.created_at};
}
let sourceReport = null;
async function refreshSources() {
  const summary=await api('sources');sourceSummary=summary;askSelections();const status=el('sourceStatus'),ledger=el('sourceLedger');ledger.replaceChildren();
  el('initSources').hidden=summary.initialized;
  if(!summary.initialized){status.textContent='Not initialized. Initialize from the curated fixture to create load 1, then append new files.';return;}
  status.textContent='Watermark load '+summary.snapshot.load_id+' · chain '+summary.snapshot.chain_hash.slice(0,12)+' · registry '+summary.registry_version+' · '+Object.entries(summary.tables).map(([t,n])=>t+' '+n+' rows').join(' · ');
  options('sourceTable',Object.keys(summary.tables).map(t=>({id:t,label:t})));
  const table=document.createElement('table'),header=table.insertRow();['Load','Table','File','Rows','Restated','Observations','Latest cutoff','Loaded','Chain'].forEach(t=>{const th=document.createElement('th');th.textContent=t;header.append(th);});
  summary.ledger.slice().reverse().forEach(e=>{const row=table.insertRow();[e.load_id,e.table_name,e.file_name,e.row_count,e.restated_rows,(e.min_observation_date||'—')+' → '+(e.max_observation_date||'—'),e.max_data_cutoff_date||'—',e.loaded_at.slice(0,19).replace('T',' '),e.chain_hash.slice(0,12)].forEach(v=>row.insertCell().textContent=String(v));});
  ledger.append(table);
}
async function refreshDocuments() {
  const data=await api('documents'),root=el('documentList');root.replaceChildren();
  el('indexDocuments').disabled=!data.embedders.length||!data.documents.some(d=>!d.indexed);
  if(!data.documents.length){root.className='empty';root.textContent=data.embedders.length?'No documents.':'No documents. No cached Qwen3-Embedding snapshot was found, so indexing is unavailable.';return;}
  root.className='';const table=document.createElement('table'),header=table.insertRow();
  ['Document','Jurisdiction','Status','Declared effective','Visible window','Superseded by','Chunks','Indexed','File'].forEach(t=>{const th=document.createElement('th');th.textContent=t;header.append(th);});
  data.documents.slice().reverse().forEach(d=>{const w=d.effective_window||{},row=table.insertRow();[d.id+(d.title?' · '+d.title:''),d.jurisdiction,d.approval_status,d.effective_from+' → '+(d.effective_to||'open'),w.effective_from+' → '+(w.effective_to||'open'),w.superseded_by||'—',d.chunk_count,d.indexed?'Yes':'No',d.file_name].forEach(v=>row.insertCell().textContent=String(v));});
  root.append(table);
}
let currentQuestion = null, sourceSummary = null, planDirty = false;
const BADGES = {new:['New · three deterministic repeats agreed',''],reused:['Reused · identical question and context',''],verified:['Verified answer',''],unstable:['Unstable · repeats disagreed, not reusable','bad'],invalid:['Invalid output · failed the answer contract','bad'],changed:['Changed since the last answer to this question','warn'],inconsistent:['Inconsistent with a similar earlier question','warn']};
function askSelections() {
  if(!state.models)return;
  ['askPlanModel','askAnswerModel'].forEach(id=>options(id,state.models));
  const adapters=task=>(state.jobs||[]).filter(j=>j.status==='completed'&&j.spec.kind==='train'&&j.spec.task===task).map(j=>({id:j.id,label:j.id.slice(0,12)+' · '+datasetLabel(j)}));
  options('askPlanAdapter',adapters('query_plan'),'Base model');options('askAnswerAdapter',adapters('credit_analysis'),'Base model');
  const versions=task=>(state.versions||[]).filter(v=>v.task===task&&v.schema.additionalProperties===false).map(v=>({id:v.id,label:versionLabel(v)}));
  options('askPlanVersion',versions('query_plan'),undefined,state.active.query_plan);options('askAnswerVersion',versions('credit_analysis'),undefined,state.active.credit_analysis);
  const loads=[...new Set((sourceSummary?.ledger||[]).map(e=>e.load_id))].sort((a,b)=>b-a);
  options('askSnapshot',loads.map((id,i)=>({id:i?String(id):'',label:(i?'Load ':'Latest · load ')+id})));
  if(!el('askAsOf').value)el('askAsOf').value=new Date().toISOString().slice(0,10);
}
function askBody(draft) {
  const choice=(model,adapter)=>({model_id:el(model).value,adapter_job_id:el(adapter).value||null,checkpoint:el('askCheckpoint').value});
  return {question:el('askQuestion').value,jurisdiction:el('askJurisdiction').value,portfolio:el('askPortfolio').value,as_of_date:el('askAsOf').value,role:el('askRole').value,snapshot_load_id:el('askSnapshot').value?Number(el('askSnapshot').value):null,plan_model:choice('askPlanModel','askPlanAdapter'),plan_version_id:el('askPlanVersion').value,answer_model:choice('askAnswerModel','askAnswerAdapter'),answer_version_id:el('askAnswerVersion').value,draft_plan:draft};
}
function planTemplate(q) {
  const asOf=new Date(q.as_of_date+'T00:00:00Z'),end=new Date(Date.UTC(asOf.getUTCFullYear(),asOf.getUTCMonth(),0)),start=new Date(Date.UTC(end.getUTCFullYear()-1,end.getUTCMonth()+1,1)),obligor=(q.question.match(/OBL[-_ ]?\d+/i)||[''])[0].toUpperCase().replace(/[-_ ]?(\d+)/,(m,n)=>'-'+n.padStart(4,'0'));
  return {portfolio:q.portfolio,jurisdiction:q.jurisdiction,entity_level:'obligor',obligor_id:obligor||'OBL-0001',date_from:start.toISOString().slice(0,10),date_to:end.toISOString().slice(0,10),as_of_date:q.as_of_date,metrics:['stage','pit_pd','days_past_due'],analysis_type:'credit_deterioration'};
}
async function refreshAsk() {
  const [questions,sessions]=await Promise.all([api('questions'),api('sessions')]);
  const history=el('askHistory');history.replaceChildren();history.className=questions.length?'':'empty';
  if(!questions.length)history.textContent='No questions yet.';
  questions.forEach(q=>{const row=document.createElement('div');row.className='row';const b=document.createElement('button');b.textContent=q.status+(q.badge?' · '+q.badge:'')+' · load '+q.snapshot.load_id+' · '+q.as_of_date+' · '+q.question.slice(0,90);b.onclick=()=>action(()=>showQuestion(q.id));row.append(b);history.append(row);});
  const root=el('askSessions');root.textContent=sessions.length?sessions.map(s=>s.status+' · '+s.model+(s.adapter_job_id?' + adapter '+s.adapter_job_id.slice(0,8):'')+' · '+s.queued_items+' waiting'+(s.releasing?' · releasing after current step':'')).join('\n'):'No model session. One starts when a step needs the model.';
}
async function showQuestion(id) {
  currentQuestion=id;const q=await api('questions/'+id);
  const waiting={drafting_plan:'The model is drafting a plan…',answering:'Generating the answer (three deterministic repeats)…',plan_ready:'Review the plan, then run it.',answered:'Answered.',failed:'Failed: '+(q.error||'see details')};
  text('askStatus',(waiting[q.status]||q.status)+' · question '+id.slice(0,8)+' · snapshot load '+q.snapshot.load_id);
  el('askPlanPanel').hidden=!['plan_ready','failed','answering','answered'].includes(q.status);
  if(!planDirty||q.id!==el('askPlan').dataset.question){const plan=q.plan_confirmed||q.plan_draft||q.plan_draft_candidate||planTemplate(q);el('askPlan').value=pretty(plan);el('askPlan').dataset.question=q.id;planDirty=false;}
  el('askRun').disabled=!['plan_ready','failed'].includes(q.status);
  el('askPlanFeedback').hidden=!q.plan_answer_id;if(q.id!==el('askPlanFeedbackResult').dataset.question){el('askPlanFeedbackResult').hidden=true;el('askPlanFeedbackResult').dataset.question=q.id;}
  text('askPlanNote',q.plan_draft?'Drafted by the plan model'+(q.plan_edited?' · you edited it':''):q.plan_error?'The model draft is not a valid plan yet: '+q.plan_error+(q.plan_draft_candidate?'\nThe draft is loaded below; fix the reported field.':(q.plan_draft_raw?'\n\nRaw draft:\n'+q.plan_draft_raw:'')+'\nA template has been filled in instead.'):'Template plan. Check the obligor, dates and metrics.');
  const panel=el('askAnswerPanel');panel.hidden=!q.answer;if(!q.answer)return q;
  const a=q.answer,badge=a.memory_status==='system'?['Policy-rule abstention · no model call','warn']:q.badge==='reused'&&a.memory_status==='invalid'?['Reused · identical inputs, but the output is invalid','bad']:(BADGES[q.badge]||[q.badge,'']);
  el('askBadge').textContent=badge[0];el('askBadge').className='badge '+badge[1];
  text('askBadgeNote',q.badge==='reused'||q.badge==='verified'?'Returned from answer memory without generating. Same plan, snapshot, documents, rules, prompt/schema, model, checkpoint and generation settings.':'Retrieval: '+a.retrieval_status+' · answered '+(a.finished_at||a.created_at).slice(0,19).replace('T',' '));
  const comparison=el('askComparison');comparison.hidden=!(q.comparison||q.precedent);if(q.comparison||q.precedent)text('askComparison',{...(q.comparison?{changed_because:q.comparison.changed_context,field_changes:q.comparison.field_changes,previous_answer_id:q.comparison.previous_answer_id}:{}),...(q.precedent?{similar_earlier_question:q.precedent.question,similarity:q.precedent.similarity,differences_from_it:q.precedent.field_changes,precedent_answer_id:q.precedent.answer_id}:{})});
  let output=a.output;try{output=typeof output==='string'?JSON.parse(output):output;}catch(_){}
  text('askOutput',output);
  text('askContext',{current_position:a.case.facts.current_position,calculated_metrics:a.case.facts.calculated_metrics,events:a.case.facts.events,evidence:a.case.evidence.map(e=>({evidence_id:e.evidence_id,score:e.score,text:e.text.slice(0,240)})),rule_evaluations:a.case.rule_evaluations,guardrail_failures:a.failures,repeat_fields:a.attempt_fields});
  text('askLineage',{snapshot:a.snapshot,documents:a.documents,keys:a.keys,sql:a.sql_lineage,identity:a.identity});
  el('askStillValid').hidden=!(q.comparison&&q.badge==='changed');el('askVerify').disabled=['unstable','invalid'].includes(a.memory_status)||q.badge==='verified';
  if(q.id!==el('askVerifyResult').dataset.question){el('askVerifyResult').hidden=true;el('askVerifyResult').dataset.question=q.id;}
  el('askFeedback').onclick=()=>action(async()=>{await refresh();document.querySelector('nav button[data-view="answers"]').click();el('answerSelect').value=a.id;inspectAnswer();});
  return q;
}
function datasetLabel(job) {return job.spec.dataset?.manifest?.name||job.spec.dataset?.manifest?.dataset_version||'No dataset';}
function renderCandidates() {
  const root=el('candidateList'),runs=(state.jobs||[]).filter(j=>j.spec.kind==='train').slice().reverse();root.replaceChildren();root.className=runs.length?'':'empty';
  if(!runs.length){root.textContent='No training runs.';return;}
  runs.forEach(j=>{const row=document.createElement('div');row.className='row';const summary=document.createElement('span');const counts=j.spec.dataset?.counts||{};const held=[counts.validation,counts.test,counts.oot].filter(Number.isFinite);const min=minReportable(),evidence=held.length===3&&Math.min(...held)>=min?'Held-out splits have at least '+min+' cases · per-metric sample labels still apply':'Small sample · fewer than '+min+' cases in a held-out split';summary.textContent=j.id.slice(0,12)+' · '+j.status+(jobDuration(j)?' · '+jobDuration(j):'')+' · '+datasetLabel(j)+' · '+evidence;row.append(summary);const checkpoint=document.createElement('select');checkpoint.setAttribute('aria-label','Checkpoint for '+j.id.slice(0,12));checkpoint.add(new Option('Best validation','best'));checkpoint.add(new Option('Final','final'));checkpoint.disabled=j.status!=='completed';row.append(checkpoint);const reference=document.createElement('select');reference.setAttribute('aria-label','Compare '+j.id.slice(0,12)+' against');reference.add(new Option('Against base model',''));runs.filter(o=>o.id!==j.id&&o.status==='completed'&&o.spec.task===j.spec.task).forEach(o=>reference.add(new Option('Against adapter '+o.id.slice(0,12),o.id)));reference.disabled=j.status!=='completed';row.append(reference);const start=document.createElement('button');start.textContent='Evaluate and compare';start.className='primary';start.disabled=j.status!=='completed';start.onclick=()=>action(async()=>{const pair=await api('comparison-runs',{training_job_id:j.id,reference_job_id:reference.value||null,checkpoint:checkpoint.value,splits:['validation','test','oot'],generation_profile:'deterministic'});selectedComparison=pair.comparison_id;await refresh();await showComparison(selectedComparison);});row.append(start);if(j.spec.task==='credit_analysis'){const replayButton=document.createElement('button');replayButton.textContent='Replay verified answers';replayButton.disabled=j.status!=='completed';replayButton.onclick=()=>action(async()=>{const job=await api('replay',{model:{model_id:j.spec.model.id,adapter_job_id:j.id,checkpoint:checkpoint.value}});await refresh();text('replayDetail',{queued_replay:job.id,answers:job.spec.answers.length});el('replayDetail').hidden=false;});row.append(replayButton);}const log=document.createElement('button');log.textContent='Download training log';log.onclick=()=>downloadJobLog(j.id,'');row.append(log);root.append(row);});
}
function renderReplays() {
  const root=el('replayList'),runs=(state.jobs||[]).filter(j=>j.spec.kind==='replay').slice().reverse();root.replaceChildren();root.className=runs.length?'':'empty';
  if(!runs.length){root.textContent='No replay runs. Confirm or correct Ask answers, then replay them against a candidate.';return;}
  runs.forEach(j=>{const row=document.createElement('div');row.className='row';const b=document.createElement('button');b.textContent=(j.spec.adapter_job_id?'adapter '+j.spec.adapter_job_id.slice(0,8):'base model')+' · '+j.status+' · '+j.spec.answers.length+' verified answers'+(jobDuration(j)?' · '+jobDuration(j):'');b.onclick=()=>action(async()=>{const d=await api('jobs/'+j.id);el('replayDetail').hidden=false;text('replayDetail',d.result?{passed:d.result.passed,matched:d.result.matched+' / '+d.result.total,mismatches:d.result.mismatches.map(m=>({question:m.question,differences:m.differences,failures:m.failures}))}:{status:d.job.status,progress:d.progress,timing:d.timing});});row.append(b);root.append(row);});
}
function renderComparisons() {
  const root=el('comparisonList'),pairs=(state.comparisons||[]).slice().reverse(),jobs=Object.fromEntries((state.jobs||[]).map(j=>[j.id,j]));root.replaceChildren();root.className=pairs.length?'':'empty';
  if(!pairs.length){root.textContent='No paired evaluations.';return;}
  pairs.forEach(p=>{const base=jobs[p.base_job_id],candidate=jobs[p.candidate_job_id],row=document.createElement('div');row.className='row';const summary=document.createElement('span');summary.textContent=p.id.slice(0,12)+' · '+(p.reference_job_id?'reference adapter '+p.reference_job_id.slice(0,8):'base')+' '+(base?.status||'missing')+' · candidate '+(candidate?.status||'missing')+' · '+p.checkpoint;row.append(summary);const view=document.createElement('button');view.textContent='View comparison';view.onclick=()=>action(()=>showComparison(p.id));row.append(view);const active=[candidate,base].filter(j=>j&&['queued','running','stopping'].includes(j.status));if(active.length){const stop=document.createElement('button');stop.textContent='Stop comparison';stop.onclick=()=>action(async()=>{for(const j of active)await api('jobs/'+j.id+'/stop',{});await refresh();if(selectedComparison===p.id)await showComparison(p.id);});row.append(stop);}root.append(row);});
}
function request() {return {task:el('task').value,kind:el('kind').value,dataset_id:el('datasetSelect').value||null,version_id:el('version').value,model_id:el('model').value,adapter_job_id:el('adapter').value||null,checkpoint:el('checkpoint').value,generation_profile:el('generationProfile').value,splits:el('splits').value.split(','),config:{epochs:Number(el('epochs').value),max_seq_length:Number(el('context').value),learning_rate:Number(el('lr').value),batch_size:Number(el('batch').value),grad_accumulation_steps:Number(el('accumulation').value),num_layers:Number(el('layers').value),target_modules:el('targetModules').value,optimizer:el('optimizer').value,weight_decay:Number(el('weightDecay').value),schedule:el('schedule').value,warmup_ratio:Number(el('warmup').value),min_lr_ratio:Number(el('minLr').value),seed:Number(el('seed').value),early_stopping_patience:Number(el('patience').value),early_stopping_min_delta:Number(el('minDelta').value),lora_parameters:{rank:Number(el('rank').value),scale:Number(el('scale').value),dropout:Number(el('dropout').value)}}};}
function draw(metrics) {
  const canvas=el('lossChart'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  const train=metrics.filter(m=>m.train).map(m=>[m.train.iteration,m.train.train_loss]);
  const valid=metrics.filter(m=>m.validation).map(m=>[m.validation.iteration,m.validation.val_loss]);
  const points=[...train,...valid].filter(p=>p.every(Number.isFinite));
  ctx.font='14px system-ui';if(!points.length){ctx.fillText('Not evaluated — no loss measurements yet.',25,40);return;}
  const maxX=Math.max(1,...points.map(p=>p[0])),maxY=Math.max(.1,...points.map(p=>p[1]));
  ctx.strokeStyle='#bdcccf';ctx.strokeRect(50,20,canvas.width-75,160);
  [[train,'#096e66','Training loss'],[valid,'#c46b20','Validation loss']].forEach(([values,color,label],i)=>{ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();values.forEach((p,n)=>{const x=50+p[0]/maxX*(canvas.width-75),y=180-p[1]/maxY*160;if(n)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();values.forEach(p=>{ctx.beginPath();ctx.arc(50+p[0]/maxX*(canvas.width-75),180-p[1]/maxY*160,3,0,Math.PI*2);ctx.fill();});ctx.fillText(label,50+i*180,210);});ctx.fillStyle='#183340';ctx.fillText('0',20,180);ctx.fillText(maxY.toFixed(2),5,25);ctx.fillText(maxX+' micro-batches',canvas.width-160,210);
}
function localProgress(d) {
  if(d.progress?.total_micro_batches||d.progress?.total_cases)return d.progress;
  if(d.job.spec.kind!=='train')return {percent:null};
  const c=d.job.spec.config,n=d.job.spec.dataset.counts.train;
  const total=Math.ceil(Math.ceil(n*c.epochs/c.batch_size)/c.grad_accumulation_steps)*c.grad_accumulation_steps;
  const seen=d.metrics.flatMap(m=>['train','validation'].filter(k=>m[k]?.iteration!==undefined).map(k=>m[k].iteration));
  const current=d.job.status==='completed'?total:Math.max(0,...seen),a=c.grad_accumulation_steps;
  return {current_micro_batches:current,total_micro_batches:total,current_optimizer_updates:Math.floor(current/a),total_optimizer_updates:Math.floor(total/a),percent:Math.round(1000*current/total)/10};
}
async function downloadJobLog(id, visibleLog) {
  let blob;
  try {const response=await fetch('/api/jobs/'+id+'/log');if(!response.ok)throw Error();blob=await response.blob();}
  catch (_) {blob=new Blob([visibleLog],{type:'text/plain'});}
  const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=id+'.log';link.click();URL.revokeObjectURL(url);
}
async function showJob(id){selectedJob=id;const d=await api('jobs/'+id);draw(d.metrics);const p=localProgress(d),bar=el('jobProgress');const timed=timingText(d.timing);if(p.percent===null){bar.removeAttribute('value');text('progressText',d.job.status+' · progress is reported when results are written'+timed);}else if(p.total_cases!==undefined){bar.value=p.percent;text('progressText',d.job.status+' · '+p.percent.toFixed(1)+'% · '+p.current_cases+'/'+p.total_cases+' cases'+timed);}else{bar.value=p.percent;text('progressText',d.job.status+' · '+p.percent.toFixed(1)+'% · '+p.current_micro_batches+'/'+p.total_micro_batches+' micro-batches · '+p.current_optimizer_updates+'/'+p.total_optimizer_updates+' optimizer updates'+timed);}text('jobDetails',{status:d.job.status,timing:d.timing,result:d.result,metrics:d.metrics.slice(-4)});text('jobLog',d.log||'Waiting for log output…');const button=el('downloadLog');button.disabled=!d.log;button.onclick=()=>downloadJobLog(id,d.log);const last=d.metrics.filter(m=>m.train).at(-1)?.train;text('runNumbers',last?pretty(last):(d.job.status==='failed'?'No optimizer update completed. See the job log below.':'Waiting for the first training report.'));}
function metricTable(parent, title, metrics, task) {
  const h=document.createElement('h3');h.textContent=title;parent.append(h);
  if(!Object.keys(metrics).length){const p=document.createElement('p');p.textContent='Not evaluated';parent.append(p);return;}
  const table=document.createElement('table');const header=table.insertRow();['Metric','Value','Denominator','95% interval','Sample'].forEach(t=>{const th=document.createElement('th');th.textContent=t;header.append(th);});
  const standard = task==='query_plan' ? ['json_validity','plan_validity','plan:obligor_id','plan:metrics','plan:date_from','plan:date_to','table_correctness','column_correctness','join_correctness','compilation_success','result_agreement','repeated_agreement','equivalent_agreement','negative_control_distinction'] : ['json_validity','numerical_agreement','extractive_support_heuristic','citation_resolution','citation_recall','driver_precision','driver_recall','abstention_recall','repeated_agreement','equivalent_agreement','negative_control_distinction'];
  const expanded=Object.fromEntries(standard.map(k=>[k,null]));Object.assign(expanded,metrics);
  Object.entries(expanded).forEach(([k,v])=>{const row=table.insertRow();const interval=v?.ci95?v.ci95.map(x=>(x*100).toFixed(1)+'%').join(' – '):'—';const label=lowerIsBetter(k)?k+' (lower is better)':k;[label,v ? (v.value*100).toFixed(1)+'%' : 'Not evaluated',v ? v.denominator : '—',interval,v ? (v.sufficient_sample??v.denominator>=minReportable()?'Reportable':'Small sample'):'—'].forEach(t=>row.insertCell().textContent=t);});parent.append(table);
}
function pairedMetricTable(parent,title,metrics,task) {
  const h=document.createElement('h3');h.textContent=title;parent.append(h);const standard=task==='query_plan'?['json_validity','plan_validity','table_correctness','column_correctness','join_correctness','compilation_success','result_agreement']:['json_validity','numerical_agreement','extractive_support_heuristic','citation_resolution','citation_recall','driver_precision','driver_recall','abstention_recall','repeated_agreement','equivalent_agreement','negative_control_distinction'];const expanded=Object.fromEntries(standard.map(k=>[k,null]));Object.assign(expanded,metrics||{});const table=document.createElement('table'),header=table.insertRow();['Metric','Base','Candidate','Delta','Denominator','Evidence state'].forEach(t=>{const th=document.createElement('th');th.textContent=t;header.append(th);});
  Object.entries(expanded).forEach(([name,value])=>{const lower=value?.lower_is_better||lowerIsBetter(name),unequal=value?.base_denominator!==undefined;let evidence='Not evaluated';if(value){if(value.base===null||value.candidate===null)evidence='Measured for one model only';else if(unequal)evidence='Unequal denominators · see paired cases';else evidence=value.denominator<minReportable()?'Small sample':'Reportable';}
    const pct=n=>Number.isFinite(n)?(n*100).toFixed(1)+'%':'—';let delta='—';if(Number.isFinite(value?.delta)){const better=lower?value.delta<0:value.delta>0;delta=(value.delta>=0?'+':'')+(value.delta*100).toFixed(1)+' pp'+(value.delta===0?'':better?' · better':' · worse');}
    const row=table.insertRow();[name+(lower?' (lower is better)':''),pct(value?.base),pct(value?.candidate),delta,value?(unequal?value.base_denominator+' base / '+value.candidate_denominator+' candidate':value.denominator):'—',evidence].forEach(t=>row.insertCell().textContent=t);});parent.append(table);
}
function releaseGates(root, runs, calibration, portfolios, task) {
  root.replaceChildren();const available=runs.filter(([,r])=>r?.release_evaluation);
  if(!available.length&&!calibration&&!portfolios){root.className='empty';root.textContent='Not evaluated.';return;}root.className='';
  if(available.length){const table=document.createElement('table'),header=table.insertRow();['Release gate',...available.map(([n])=>n)].forEach(t=>{const th=document.createElement('th');th.textContent=t;header.append(th);});
    const rows=[['Status',g=>g.status],['Passed',g=>g.passed?'Yes':'No'],['Promotable',g=>g.promotable?'Yes':'No'],['Gate checks',g=>g.checked],['Failures',g=>(g.failures||[]).length],['Insufficient evidence',g=>(g.insufficient_evidence||[]).length],['Coverage missing',g=>[...(g.coverage_missing||[]),...(g.metric_coverage_missing||[])].join(', ')||'None'],['Cases scored',(g,r)=>r.overall?.n??'—']];
    rows.forEach(([label,f])=>{const row=table.insertRow();row.insertCell().textContent=label;available.forEach(([,r])=>row.insertCell().textContent=String(f(r.release_evaluation.gates||{},r.release_evaluation)));});root.append(table);
    available.forEach(([n,r])=>{const g=r.release_evaluation.gates||{};detail(root,n+' gate details',{failures:g.failures,blocking_failures:g.blocking_failures,insufficient_evidence:g.insufficient_evidence,promotion_block:g.promotion_block,overall:r.release_evaluation.overall,by_portfolio:r.release_evaluation.by_portfolio});});}
  else{const p=document.createElement('p');p.className='muted';p.textContent='Release gates not evaluated: no test or OOT credit-analysis cases in this run.';root.append(p);}
  if(calibration)detail(root,'Calibration by split',calibration);
  if(portfolios)Object.entries(portfolios).filter(([,m])=>Object.keys(m).length).forEach(([name,m])=>pairedMetricTable(root,'Portfolio · '+name,m,task));
}
async function showComparison(id) {
  selectedComparison=id;const d=await api('comparison-runs/'+id),base=d.base,candidate=d.candidate,bp=base.progress||{},cp=candidate.progress||{};const values=[bp.percent,cp.percent].filter(Number.isFinite),percent=values.length?values.reduce((a,b)=>a+b,0)/2:0,bar=el('comparisonProgress');bar.value=percent;const side=(name,job,progress,t)=>name+' '+job.status+(progress.total_cases?' '+progress.current_cases+'/'+progress.total_cases:'')+timingText(job.status==='queued'?null:t);text('comparisonProgressText',d.status+' · '+percent.toFixed(1)+'% overall'+(Number.isFinite(d.timing?.eta_seconds)?' · about '+duration(d.timing.eta_seconds)+' left overall':'')+(d.status==='completed'&&Number.isFinite(d.timing?.elapsed_seconds)?' · took '+duration(d.timing.elapsed_seconds):'')+'\n'+side('Base',base.job,bp,base.timing)+'\n'+side('Candidate',candidate.job,cp,candidate.timing));text('comparisonSummary',{comparison_id:id,training_job_id:d.comparison.training_job_id,checkpoint:d.comparison.checkpoint,splits:d.comparison.splits,base_job_id:base.job.id,candidate_job_id:candidate.job.id,compatibility:d.comparison.compatibility,compatibility_error:d.compatibility_error});const root=el('scorecards'),cases=el('evalCases'),gates=el('releaseGates');root.replaceChildren();cases.replaceChildren();gates.replaceChildren();if(!d.metrics){const message=d.status==='failed'?'Comparison failed. Inspect the base and candidate job logs.':d.status==='incompatible'?'Comparison incompatible: '+d.compatibility_error:'Results will appear after both evaluations complete.';[root,cases,gates].forEach(n=>{n.className='empty';n.textContent=message;});return;}root.className='';const task=candidate.job.spec.task;Object.entries(d.metrics.scorecards).filter(([,m])=>Object.keys(m).length).forEach(([split,metrics])=>pairedMetricTable(root,split,metrics,task));detail(root,'Paired case wins, losses and ties',d.metrics.paired);releaseGates(gates,[['Base',base.result],['Candidate',candidate.result]],{base:base.result?.calibration,candidate:candidate.result?.calibration},d.metrics.portfolios,task);if(d.consistency_gates){const h=document.createElement('h3');h.textContent='Consistency gates · '+d.consistency_gates.status;gates.prepend(h);const table=document.createElement('table'),header=table.insertRow();['Gate','Scope','Value','Threshold','Status'].forEach(t=>{const th=document.createElement('th');th.textContent=t;header.append(th);});d.consistency_gates.results.forEach(r=>{const row=table.insertRow();[r.gate,r.scope,r.value===undefined?'—':(r.gate==='regression_checks'?r.value+' newly failing':(r.value*100).toFixed(1)+'%'+(r.denominator?' of '+r.denominator:'')),r.minimum!==undefined?'≥ '+(r.minimum*100).toFixed(0)+'%':(r.maximum_newly_failing!==undefined?'≤ '+r.maximum_newly_failing:'—'),r.status.replaceAll('_',' ')+(r.note?' · '+r.note:'')].forEach(v=>row.insertCell().textContent=String(v));});h.after(table);}cases.className='';const baseRows=Object.fromEntries((base.result?.cases||[]).map(c=>[c.case_id,c]));(candidate.result?.cases||[]).forEach(c=>{const b=baseRows[c.case_id];detail(cases,c.case_id+' · '+c.split+' · base '+(b?b.failures.length:'—')+' failures · candidate '+c.failures.length+' failures',{base:b||null,candidate:c});});
}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('main > section').forEach(s=>s.hidden=s.id!==b.dataset.view);document.querySelectorAll('nav button').forEach(n=>n.classList.toggle('active',n===b));});
el('task').onchange=selections;el('version').onchange=selections;
el('context').value='1664';
for(const value of ['8','4','1'])if(![...el('layers').options].some(o=>o.value===value))el('layers').add(new Option(value,value),el('layers').firstChild);
el('layers').value='1';el('targetModules').value='attention';el('rank').value='8';el('dropout').value='0';
el('initSources').onclick=()=>action(async()=>{text('sourceReport',await api('sources/initialize',{}));await refreshSources();});
el('validateSource').onclick=()=>action(async()=>{const file=el('sourceFile').files[0];if(!file)throw Error('Choose a Parquet, CSV or JSONL file');el('appendSource').disabled=true;sourceReport=await upload('sources/stage?table='+encodeURIComponent(el('sourceTable').value)+'&filename='+encodeURIComponent(file.name),file);text('sourceReport',{passed:sourceReport.passed,table:sourceReport.table,rows:sourceReport.rows,restated_rows:sourceReport.restated_rows,observations:sourceReport.min_observation_date+' → '+sourceReport.max_observation_date,latest_cutoff:sourceReport.max_data_cutoff_date,errors:sourceReport.errors,note:sourceReport.passed?'Valid. Append creates the next load; restated rows supersede earlier versions only for later snapshots.':'Nothing can be appended until every error is fixed.'});el('appendSource').disabled=!sourceReport.passed;});
el('appendSource').onclick=()=>action(async()=>{if(!sourceReport?.passed)throw Error('Validate a file first');const entry=await api('sources/append',{staged_id:sourceReport.staged_id,table:sourceReport.table,file_name:sourceReport.file_name});sourceReport=null;el('appendSource').disabled=true;el('sourceFile').value='';text('sourceReport',{appended:entry});await refreshSources();});
el('registerDocument').onclick=()=>action(async()=>{const file=el('documentFile').files[0];if(!file)throw Error('Choose a PDF, DOCX, Markdown or text file');if(!el('documentFrom').value)throw Error('Set the effective-from date');const staged=await upload('documents/stage?filename='+encodeURIComponent(file.name),file);const metadata={document_id:el('documentId').value.trim(),version:el('documentVersion').value.trim(),jurisdiction:el('documentJurisdiction').value,title:el('documentTitle').value.trim(),document_type:el('documentType').value.trim()||'policy',approval_status:el('documentStatus').value,effective_from:el('documentFrom').value,effective_to:el('documentTo').value||null,supersedes_document_id:el('documentSupersedes').value.trim()||null,confidentiality_level:el('documentConfidentiality').value,portfolio:el('documentPortfolio').value.split(',').map(p=>p.trim()).filter(Boolean)};const record=await api('documents',{staged_id:staged.staged_id,file_name:staged.file_name,metadata});text('documentReport',{registered:record.id,chunks:record.chunk_count,visible_window:record.effective_window,next:'Choose Index documents to embed it for retrieval.'});el('documentFile').value='';await refreshDocuments();});
el('indexDocuments').onclick=()=>action(async()=>{const job=await api('documents/index',{});text('documentReport',{queued_indexing_job:job.id,documents:job.spec.documents,note:'Runs in the single model lane; progress appears in Runs.'});await refresh();await refreshDocuments();});
el('replayBase').onclick=()=>action(async()=>{const model=state.models?.[0];if(!model)throw Error('No cached base model');const job=await api('replay',{model:{model_id:model.id}});await refresh();el('replayDetail').hidden=false;text('replayDetail',{queued_replay:job.id,answers:job.spec.answers.length});});
let builtManifest=null;
el('buildDataset').onclick=()=>action(async()=>{let paraphrases={};if(el('buildParaphrases').value.trim()){try{paraphrases=JSON.parse(el('buildParaphrases').value);}catch(_){throw Error('Paraphrases must be JSON: {"feedback-id": ["question", ...]}');}}const result=await api('datasets/build',{base_dataset_id:el('buildBase').value,dataset_version:el('buildVersion').value.trim(),feedback_fraction:Number(el('buildFraction').value),paraphrases});builtManifest=result.manifest_path;el('registerBuilt').disabled=false;text('buildResult',result);});
el('registerBuilt').onclick=()=>action(async()=>{if(!builtManifest)throw Error('Build a dataset version first');await api('datasets',{path:builtManifest});el('registerBuilt').disabled=true;text('buildResult','Registered '+builtManifest+'. Preflight and train it from Runs.');await refresh();});
el('askPlanFeedback').onclick=()=>action(async()=>{const comment=prompt('What was wrong with the drafted plan? (optional)')||'';const record=await api('questions/'+currentQuestion+'/plan-feedback',{comment});el('askPlanFeedbackResult').hidden=false;text('askPlanFeedbackResult',{explanation:ELIGIBILITY[record.eligibility_status]||null,eligibility:record.eligibility_status,regression:record.regression_status});await refresh();});
el('askVerify').onclick=()=>action(async()=>{const q=await api('questions/'+currentQuestion);const result=await api('answers/'+q.answer_id+'/verify',{comment:''});el('askVerifyResult').hidden=false;text('askVerifyResult',result);await refresh();});
el('askStillValid').onclick=()=>action(async()=>{await api('questions/'+currentQuestion+'/still-valid',{comment:''});await refresh();await showQuestion(currentQuestion);});
el('askPlan').oninput=()=>{planDirty=true;};
el('askDraft').onclick=()=>action(async()=>{planDirty=false;const q=await api('questions',askBody(true));await refreshAsk();await showQuestion(q.id);});
el('askManual').onclick=()=>action(async()=>{planDirty=false;const q=await api('questions',askBody(false));await refreshAsk();await showQuestion(q.id);});
el('askRun').onclick=()=>action(async()=>{if(!currentQuestion)throw Error('Ask a question first');let plan;try{plan=JSON.parse(el('askPlan').value);}catch(_){throw Error('The plan is not valid JSON');}planDirty=false;await api('questions/'+currentQuestion+'/plan',{plan});await refresh();await refreshAsk();await showQuestion(currentQuestion);});
el('askRelease').onclick=()=>action(async()=>{text('askStatus',await api('sessions/release',{}));await refresh();await refreshAsk();});
el('register').onclick=()=>action(async()=>{await api('datasets',{path:el('datasetPath').value});await refresh();});
el('preflight').onclick=()=>action(async()=>{text('preflightResult',await api('preflight',request()));await refresh();});
el('start').onclick=()=>action(async()=>{const j=await api('jobs',request());await refresh();await showJob(j.id);});
el('loadEvaluation').onclick=()=>action(async()=>{if(!el('evalRun').value)throw Error('Select an evaluation run');const d=await api('jobs/'+el('evalRun').value);selectedComparison=null;text('comparisonProgressText','Showing a standalone evaluation, not a paired comparison.');el('comparisonProgress').value=0;text('comparisonSummary',{evaluation_job_id:d.job.id,kind:d.job.spec.kind,task:d.job.spec.task,splits:d.job.spec.splits,identity:d.result?.identity||null});const root=el('scorecards'),cases=el('evalCases'),gates=el('releaseGates');[root,cases,gates].forEach(n=>{n.replaceChildren();n.className='';});if(!d.result?.splits){[root,cases,gates].forEach(n=>{n.className='empty';n.textContent='Not evaluated';});return;}Object.entries(d.result.splits).filter(([,m])=>Object.keys(m).length).forEach(([s,m])=>metricTable(root,s==='development'?'Development checks — not held-out accuracy':s,m,d.job.spec.task));releaseGates(gates,[['Run',d.result]],d.result.calibration,null,d.job.spec.task);Object.entries(d.result.portfolios||{}).filter(([,m])=>Object.keys(m).length).forEach(([p,m])=>metricTable(gates,'Portfolio · '+p,m,d.job.spec.task));d.result.cases.forEach(c=>detail(cases,c.case_id+' · '+c.split+' · '+c.failures.length+' failures',c));});
el('compare').onclick=()=>action(async()=>text('comparison',await api('compare/'+el('compareBase').value+'/'+el('compareCandidate').value+'?mode='+el('compareMode').value)));
function inspectAnswer() {
  const a=state.answers.find(a=>a.id===el('answerSelect').value);inspectedAnswer=a?.id||null;
  if(!a){text('answerInput','No answer selected.');text('answerOutput','No answer selected.');text('sqlView','SQL lineage unavailable.');return;}
  text('answerInput',a.case);text('answerOutput',{source:answerLabel(a),output:a.output,metrics:a.metrics,failures:a.failures,version_id:a.version_id,job_id:a.job_id,comparison_id:a.comparison_id,comparison_role:a.comparison_role,identity:a.identity});text('sqlView',a.sql_lineage||'SQL lineage unavailable for this case.');el('feedbackTask').value=a.case.task;
}
el('answerSelect').onchange=inspectAnswer;
el('loadAnswer').onclick=()=>action(async()=>{if(!el('answerSelect').value)throw Error('Select an answer');inspectAnswer();});
el('submitFeedback').onclick=()=>action(async()=>{if(!inspectedAnswer||inspectedAnswer!==el('answerSelect').value)throw Error('Select and inspect an answer before submitting feedback');const correction=el('correction').value.trim();const result=await api('feedback',{submission_id:crypto.randomUUID(),interaction_id:inspectedAnswer,comment:el('feedbackComment').value,correction:correction||null,cause:el('cause').value,expectations:el('expectations').value.trim()?JSON.parse(el('expectations').value):null});text('feedbackResult',{explanation:ELIGIBILITY[result.eligibility_status]||null,...result});await refresh();});
el('exportBatch').onclick=()=>action(async()=>{const data=await api('feedback/batch/'+el('feedbackTask').value);text('feedbackResult',{task:data.task,eligible_cases:data.cases.length,skipped_reasons:data.skipped_reasons||{skipped:data.skipped.length},duplicate_or_capped:data.duplicate_or_capped.length,note:data.cases.length?data.note:'No eligible feedback for this task, so nothing was downloaded. '+ELIGIBILITY.protected_split_or_group});if(!data.cases.length)return;const url=URL.createObjectURL(new Blob([pretty(data)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='feedback-fragment.json';link.click();URL.revokeObjectURL(url);});
el('importAnswer').onclick=()=>action(async()=>{await api('answers',JSON.parse(el('importPayload').value));await refresh();});
el('loadVersion').onclick=()=>action(async()=>{const v=state.versions.find(v=>v.id===el('editVersion').value);el('promptEdit').value=v.prompt;el('schemaEdit').value=pretty(v.schema);el('versionName').value=v.name+' — edited';});
el('saveVersion').onclick=()=>action(async()=>{const v=state.versions.find(v=>v.id===el('editVersion').value);await api('versions',{task:v.task,prompt:el('promptEdit').value,schema:JSON.parse(el('schemaEdit').value),name:el('versionName').value,parent:v.id});await refresh();});
el('activateVersion').onclick=()=>action(async()=>{await api('versions/'+el('editVersion').value+'/activate',{});await refresh();});
action(async()=>{token=(await api('session')).token;await refresh();await refreshSources().catch(e=>text('sourceStatus',e.message));await refreshDocuments().catch(e=>text('documentReport',e.message));await refreshAsk().catch(e=>text('askStatus',e.message));});
setInterval(async()=>{try{if(state.jobs?.some(j=>['running','queued','stopping'].includes(j.status))){await refresh();if(selectedJob)await showJob(selectedJob);if(selectedComparison)await showComparison(selectedComparison);if(!el('datasets').hidden)await refreshDocuments();}if(currentQuestion&&!el('ask').hidden){const q=await showQuestion(currentQuestion);if(q&&['drafting_plan','answering'].includes(q.status)||state.jobs?.some(j=>j.spec.kind==='session'&&['running','queued'].includes(j.status)))await refreshAsk();}}catch(e){text('status',e.message);}},3000);
