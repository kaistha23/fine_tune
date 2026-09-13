'use strict';
let state = {}, token = '', selectedJob = null, selectedComparison = null;
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
function selections() {
  const task=el('task').value;
  options('datasetSelect',(state.datasets||[]).filter(d=>d.manifest.task===task).map(d=>({id:d.id,label:d.manifest.name||d.id.slice(0,12)})),'Select registered data');
  options('model',(state.models||[]));
  options('version',(state.versions||[]).filter(v=>v.task===task&&v.schema.additionalProperties===false).map(v=>({id:v.id,label:v.name+(state.active[task]===v.id?' · active':'')})),undefined,state.active[task]);
  options('adapter',(state.jobs||[]).filter(j=>j.status==='completed'&&j.spec.kind==='train'&&j.spec.task===task).map(j=>({id:j.id,label:j.id.slice(0,12)})),'Base model');
}
async function refresh() {
  state=await api('state');selections();
  options('answerSelect',state.answers.map(a=>({id:a.id,label:a.case.question.slice(0,100)})),'Select an answer');
  options('editVersion',state.versions.map(v=>({id:v.id,label:v.task+' / '+v.name})));
  const evaluations=state.jobs.filter(j=>j.status==='completed'&&j.spec.kind==='evaluate').map(j=>({id:j.id,label:(j.spec.comparison_role?j.spec.comparison_role+' / ':'')+j.spec.task+' / '+j.id.slice(0,12)}));
  ['evalRun','compareBase','compareCandidate'].forEach(id=>options(id,evaluations,'Select evaluation'));
  renderCandidates();renderComparisons();
  const list=el('datasetList');list.replaceChildren();
  if(!state.datasets.length){list.className='empty';list.textContent='No datasets registered. Phase-2 data can be added later.';}else{list.className='';state.datasets.forEach(d=>detail(list,d.manifest.name||d.id.slice(0,12),{task:d.manifest.task,contract_version:d.contract_version,contract_warnings:d.contract_warnings,counts:d.counts,taxonomy_counts:d.taxonomy_counts,diversity:d.diversity,missing_targets:d.missing_targets,missing_expectations:d.missing_expectations,token_lengths:d.token_lengths||'Run preflight to measure',hash:d.hash}));}
  const jobs=el('jobList');jobs.replaceChildren();
  if(!state.jobs.length)jobs.textContent='No runs. Training and evaluation are Not evaluated.';
  state.jobs.slice().reverse().forEach(j=>{const row=document.createElement('div');row.className='row';const b=document.createElement('button');b.textContent=j.spec.task+' · '+j.spec.kind+' · '+j.status+' · '+j.id.slice(0,8);b.onclick=()=>action(()=>showJob(j.id));row.append(b);if(['queued','running','stopping'].includes(j.status)){const stop=document.createElement('button');stop.textContent='Stop';stop.onclick=()=>action(async()=>{await api('jobs/'+j.id+'/stop',{});await refresh();});row.append(stop);}jobs.append(row);});
  const history=el('feedbackList');history.replaceChildren();history.className=state.feedback.length?'':'empty';
  if(!state.feedback.length)history.textContent='No feedback submitted.';
  state.feedback.slice().reverse().forEach(f=>detail(history,f.cause+' · '+(f.eligibility_status||f.regression_status)+' · '+f.created_at,{comment:f.comment,diagnostics:f.diagnostics,has_independent_expectations:f.has_independent_expectations,eligible_for_training:f.eligible_for_training,protected_split_or_group:f.protected_split_or_group,recommendations:f.recommendations}));
}
function datasetLabel(job) {return job.spec.dataset?.manifest?.name||job.spec.dataset?.manifest?.dataset_version||'No dataset';}
function renderCandidates() {
  const root=el('candidateList'),runs=(state.jobs||[]).filter(j=>j.spec.kind==='train').slice().reverse();root.replaceChildren();root.className=runs.length?'':'empty';
  if(!runs.length){root.textContent='No training runs.';return;}
  runs.forEach(j=>{const row=document.createElement('div');row.className='row';const summary=document.createElement('span');const counts=j.spec.dataset?.counts||{};const held=[counts.validation,counts.test,counts.oot].filter(Number.isFinite);const evidence=held.length===3&&Math.min(...held)>=30?'Coverage requires metric eligibility checks':'Mechanics only · fewer than 30 cases in a held-out split';summary.textContent=j.id.slice(0,12)+' · '+j.status+' · '+datasetLabel(j)+' · '+evidence;row.append(summary);const checkpoint=document.createElement('select');checkpoint.setAttribute('aria-label','Checkpoint for '+j.id.slice(0,12));checkpoint.add(new Option('Best validation','best'));checkpoint.add(new Option('Final','final'));checkpoint.disabled=j.status!=='completed';row.append(checkpoint);const start=document.createElement('button');start.textContent='Evaluate and compare';start.className='primary';start.disabled=j.status!=='completed';start.onclick=()=>action(async()=>{const pair=await api('comparison-runs',{training_job_id:j.id,checkpoint:checkpoint.value,splits:['validation','test','oot'],generation_profile:'deterministic'});selectedComparison=pair.comparison_id;await refresh();await showComparison(selectedComparison);});row.append(start);const log=document.createElement('button');log.textContent='Download training log';log.onclick=()=>downloadJobLog(j.id,'');row.append(log);root.append(row);});
}
function renderComparisons() {
  const root=el('comparisonList'),pairs=(state.comparisons||[]).slice().reverse(),jobs=Object.fromEntries((state.jobs||[]).map(j=>[j.id,j]));root.replaceChildren();root.className=pairs.length?'':'empty';
  if(!pairs.length){root.textContent='No paired evaluations.';return;}
  pairs.forEach(p=>{const base=jobs[p.base_job_id],candidate=jobs[p.candidate_job_id],row=document.createElement('div');row.className='row';const summary=document.createElement('span');summary.textContent=p.id.slice(0,12)+' · base '+(base?.status||'missing')+' · candidate '+(candidate?.status||'missing')+' · '+p.checkpoint;row.append(summary);const view=document.createElement('button');view.textContent='View comparison';view.onclick=()=>action(()=>showComparison(p.id));row.append(view);root.append(row);});
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
  if(d.progress?.total_micro_batches)return d.progress;
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
async function showJob(id){selectedJob=id;const d=await api('jobs/'+id);draw(d.metrics);const p=localProgress(d),bar=el('jobProgress');if(p.percent===null){bar.removeAttribute('value');text('progressText',d.job.status+' · progress is reported when results are written');}else{bar.value=p.percent;text('progressText',d.job.status+' · '+p.percent.toFixed(1)+'% · '+p.current_micro_batches+'/'+p.total_micro_batches+' micro-batches · '+p.current_optimizer_updates+'/'+p.total_optimizer_updates+' optimizer updates');}text('jobDetails',{status:d.job.status,result:d.result,metrics:d.metrics.slice(-4)});text('jobLog',d.log||'Waiting for log output…');const button=el('downloadLog');button.disabled=!d.log;button.onclick=()=>downloadJobLog(id,d.log);const last=d.metrics.filter(m=>m.train).at(-1)?.train;text('runNumbers',last?pretty(last):(d.job.status==='failed'?'No optimizer update completed. See the job log below.':'Waiting for the first training report.'));}
function metricTable(parent, title, metrics, task) {
  const h=document.createElement('h3');h.textContent=title;parent.append(h);
  if(!Object.keys(metrics).length){const p=document.createElement('p');p.textContent='Not evaluated';parent.append(p);return;}
  const table=document.createElement('table');const header=table.insertRow();['Metric','Value','Denominator','95% interval','Sample'].forEach(t=>{const th=document.createElement('th');th.textContent=t;header.append(th);});
  const standard = task==='query_plan' ? ['json_validity','plan_validity','plan:obligor_id','plan:metrics','plan:date_from','plan:date_to','table_correctness','column_correctness','join_correctness','compilation_success','result_agreement','repeated_agreement','equivalent_agreement','negative_control_distinction'] : ['json_validity','numerical_agreement','extractive_support_heuristic','citation_resolution','citation_recall','driver_precision','driver_recall','abstention_recall','repeated_agreement','equivalent_agreement','negative_control_distinction'];
  const expanded=Object.fromEntries(standard.map(k=>[k,null]));Object.assign(expanded,metrics);
  Object.entries(expanded).forEach(([k,v])=>{const row=table.insertRow();const interval=v?.ci95?v.ci95.map(x=>(x*100).toFixed(1)+'%').join(' – '):'—';const label=k==='confidence_brier_score'?k+' (lower is better)':k;[label,v ? (v.value*100).toFixed(1)+'%' : 'Not evaluated',v ? v.denominator : '—',interval,v ? (v.sufficient_sample?'Reportable':'Small sample'):'—'].forEach(t=>row.insertCell().textContent=t);});parent.append(table);
}
function pairedMetricTable(parent,title,metrics,task) {
  const h=document.createElement('h3');h.textContent=title;parent.append(h);const standard=task==='query_plan'?['json_validity','plan_validity','table_correctness','column_correctness','join_correctness','compilation_success','result_agreement']:['json_validity','numerical_agreement','extractive_support_heuristic','citation_resolution','citation_recall','driver_precision','driver_recall','abstention_recall','repeated_agreement','equivalent_agreement','negative_control_distinction'];const expanded=Object.fromEntries(standard.map(k=>[k,null]));Object.assign(expanded,metrics||{});const table=document.createElement('table'),header=table.insertRow();['Metric','Base','Candidate','Delta','Denominator','Evidence state'].forEach(t=>{const th=document.createElement('th');th.textContent=t;header.append(th);});Object.entries(expanded).forEach(([name,value])=>{const denominator=value?.denominator;let evidence='Not evaluated';if(value){evidence=denominator<10?'Small sample':denominator<30?'Insufficient evidence to gate':'Measured';}const pct=n=>Number.isFinite(n)?(n*100).toFixed(1)+'%':'—',row=table.insertRow();[name,pct(value?.base),pct(value?.candidate),Number.isFinite(value?.delta)?((value.delta>=0?'+':'')+(value.delta*100).toFixed(1)+' pp'):'—',denominator??'—',evidence].forEach(t=>row.insertCell().textContent=t);});parent.append(table);
}
async function showComparison(id) {
  selectedComparison=id;const d=await api('comparison-runs/'+id),base=d.base,candidate=d.candidate,bp=base.progress||{},cp=candidate.progress||{};const values=[bp.percent,cp.percent].filter(Number.isFinite),percent=values.length?values.reduce((a,b)=>a+b,0)/2:0,bar=el('comparisonProgress');bar.value=percent;text('comparisonProgressText',d.status+' · Base '+base.job.status+(bp.total_cases?' '+bp.current_cases+'/'+bp.total_cases:'')+' · Candidate '+candidate.job.status+(cp.total_cases?' '+cp.current_cases+'/'+cp.total_cases:'')+' · '+percent.toFixed(1)+'% overall');text('comparisonSummary',{comparison_id:id,training_job_id:d.comparison.training_job_id,checkpoint:d.comparison.checkpoint,splits:d.comparison.splits,base_job_id:base.job.id,candidate_job_id:candidate.job.id,compatibility:d.comparison.compatibility,compatibility_error:d.compatibility_error});const root=el('scorecards');root.replaceChildren();if(!d.metrics){root.className='empty';root.textContent=d.status==='failed'?'Comparison failed. Inspect the base and candidate job logs.':'Scorecards will appear after both evaluations complete.';return;}root.className='';Object.entries(d.metrics.scorecards).forEach(([split,metrics])=>pairedMetricTable(root,split,metrics,candidate.job.spec.task));const paired=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent='Paired case wins, losses and ties';pre.textContent=pretty(d.metrics.paired);paired.append(summary,pre);root.append(paired);const cases=el('evalCases');cases.replaceChildren();cases.className='';(candidate.result?.cases||[]).forEach(c=>detail(cases,c.case_id+' · '+c.split+' · '+c.failures.length+' failures',c));
}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('main > section').forEach(s=>s.hidden=s.id!==b.dataset.view);document.querySelectorAll('nav button').forEach(n=>n.classList.toggle('active',n===b));});
el('task').onchange=selections;
el('context').value='1664';
for(const value of ['8','4','1'])if(![...el('layers').options].some(o=>o.value===value))el('layers').add(new Option(value,value),el('layers').firstChild);
el('layers').value='1';el('targetModules').value='attention';el('rank').value='8';el('dropout').value='0';
el('register').onclick=()=>action(async()=>{await api('datasets',{path:el('datasetPath').value});await refresh();});
el('preflight').onclick=()=>action(async()=>text('preflightResult',await api('preflight',request())));
el('start').onclick=()=>action(async()=>{const j=await api('jobs',request());await refresh();await showJob(j.id);});
el('loadEvaluation').onclick=()=>action(async()=>{const d=await api('jobs/'+el('evalRun').value);const root=el('scorecards');root.replaceChildren();root.className='';if(!d.result?.splits){root.textContent='Not evaluated';return;}Object.entries(d.result.splits).forEach(([s,m])=>metricTable(root,s==='development'?'Development checks — not held-out accuracy':s,m,d.job.spec.task));const cases=el('evalCases');cases.replaceChildren();cases.className='';d.result.cases.forEach(c=>detail(cases,c.case_id+' · '+c.split+' · '+c.failures.length+' failures',c));});
el('compare').onclick=()=>action(async()=>text('comparison',await api('compare/'+el('compareBase').value+'/'+el('compareCandidate').value+'?mode='+el('compareMode').value)));
el('loadAnswer').onclick=()=>action(async()=>{const a=state.answers.find(a=>a.id===el('answerSelect').value);if(!a)throw Error('Select an answer');text('answerInput',a.case);text('answerOutput',{output:a.output,identity:a.identity,version_id:a.version_id});text('sqlView',a.sql_lineage||'SQL lineage unavailable for this case.');});
el('submitFeedback').onclick=()=>action(async()=>{const correction=el('correction').value.trim();const result=await api('feedback',{submission_id:crypto.randomUUID(),interaction_id:el('answerSelect').value,comment:el('feedbackComment').value,correction:correction||null,cause:el('cause').value,expectations:el('expectations').value.trim()?JSON.parse(el('expectations').value):null});text('feedbackResult',result);await refresh();});
el('exportBatch').onclick=()=>action(async()=>{const data=await api('feedback/batch/'+el('task').value);const url=URL.createObjectURL(new Blob([pretty(data)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='feedback-fragment.json';link.click();URL.revokeObjectURL(url);});
el('importAnswer').onclick=()=>action(async()=>{await api('answers',JSON.parse(el('importPayload').value));await refresh();});
el('loadVersion').onclick=()=>action(async()=>{const v=state.versions.find(v=>v.id===el('editVersion').value);el('promptEdit').value=v.prompt;el('schemaEdit').value=pretty(v.schema);el('versionName').value=v.name+' — edited';});
el('saveVersion').onclick=()=>action(async()=>{const v=state.versions.find(v=>v.id===el('editVersion').value);await api('versions',{task:v.task,prompt:el('promptEdit').value,schema:JSON.parse(el('schemaEdit').value),name:el('versionName').value,parent:v.id});await refresh();});
el('activateVersion').onclick=()=>action(async()=>{await api('versions/'+el('editVersion').value+'/activate',{});await refresh();});
action(async()=>{token=(await api('session')).token;await refresh();});
setInterval(async()=>{try{if(state.jobs?.some(j=>['running','queued','stopping'].includes(j.status))){await refresh();if(selectedJob)await showJob(selectedJob);if(selectedComparison)await showComparison(selectedComparison);}}catch(e){text('status',e.message);}},3000);
