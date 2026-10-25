import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const $=id=>document.getElementById(id);
const viewer=$('viewer');
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(49.1,1,.03,30);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.outputColorSpace=THREE.SRGBColorSpace;
viewer.appendChild(renderer.domElement);
renderer.domElement.setAttribute('aria-label','Reconstructed desk scene');
renderer.domElement.setAttribute('role','img');
const controls=new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true;controls.dampingFactor=.09;
controls.minDistance=.15;controls.maxDistance=7;
let meta,positions,baseColors,points,highlight,currentResult,selectedCandidate=-1,activePhoto=42;
let busy=false,ready=false,requestId=0;
let reconstructionMeta,neuralPoints,neuralVisible=false;
document.querySelectorAll('[data-query]').forEach(button=>button.disabled=true);
const color=new THREE.Color();
const resize=()=>{const w=viewer.clientWidth,h=viewer.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();};
new ResizeObserver(resize).observe(viewer);
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
controls.addEventListener('start',()=>{$('view-label').textContent='Free orbit · anchored scene points';});

function setView(frameId){
  const f=meta.frames.find(x=>x.id===frameId),m=f.c2w;
  camera.position.set(m[0][3],m[1][3],m[2][3]);
  camera.up.set(-m[0][1],-m[1][1],-m[2][1]);
  const forward=new THREE.Vector3(m[0][2],m[1][2],m[2][2]);
  const center=currentResult?.candidates[selectedCandidate]?.center;
  let distance=1.3;
  if(center)distance=THREE.MathUtils.clamp(new THREE.Vector3(...center).sub(camera.position).dot(forward),.6,2.5);
  controls.target.copy(camera.position).addScaledVector(forward,distance);
  camera.lookAt(controls.target);controls.update();
  $('view-label').textContent=`Source viewpoint ${frameId}`;
}

function showPhotos(evidence){
  $('photos').replaceChildren();
  if(!evidence.length){
    const p=document.createElement('p');p.className='empty-evidence';p.textContent='No supported source region. Try a specific visible object.';$('photos').append(p);return;
  }
  for(const e of evidence){
    const button=document.createElement('button');button.className='photo-card';
    button.setAttribute('aria-label',`Inspect source photograph ${e.frame_id}`);
    const img=document.createElement('img');img.src=`/api/photo/${e.frame_id}`;img.alt=`Source frame ${e.frame_id}`;
    button.append(img);
    if(e.box){
      const box=document.createElement('span');box.className='source-box';
      box.style.left=`${e.box[0]/640*100}%`;box.style.top=`${e.box[1]/480*108}px`;
      box.style.width=`${(e.box[2]-e.box[0])/640*100}%`;box.style.height=`${(e.box[3]-e.box[1])/480*108}px`;
      button.append(box);
    }
    const caption=document.createElement('span');caption.textContent=e.score?`Frame ${String(e.frame_id).padStart(2,'0')} · score ${e.score.toFixed(2)}`:`View ${String(e.frame_id).padStart(2,'0')} · Original RGB`;
    button.append(caption);button.onclick=()=>openPhoto(e);$('photos').append(button);
  }
}

function openPhoto(e){
  activePhoto=e.frame_id;
  const f=meta.frames.find(x=>x.id===e.frame_id);
  $('photo-title').textContent=`Source photograph / Frame ${String(e.frame_id).padStart(2,'0')}`;
  $('full-photo').src=`/api/photo/${e.frame_id}`;
  $('photo-caption').textContent=`${f.rgb} · ${f.split} view · Original, unmodified photograph. TUM RGB-D, CC BY 4.0.`;
  $('photo-dialog').showModal();
}

function clearHighlight(){
  if(highlight){scene.remove(highlight);highlight.geometry.dispose();highlight.material.dispose();highlight=null;}
  if(points)points.material.opacity=1;
  $('selection-info').textContent='No selection';
  document.body.dataset.selectionCount='0';
  document.body.dataset.selectionFingerprint='';
}

function selectCandidate(index){
  selectedCandidate=index;clearHighlight();
  const c=currentResult.candidates[index];
  const values=new Float32Array(c.point_ids.length*3);
  c.point_ids.forEach((id,j)=>values.set(positions.subarray(id*3,id*3+3),j*3));
  const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.BufferAttribute(values,3));
  const mat=new THREE.PointsMaterial({color:0xb6ffd1,size:.014,sizeAttenuation:true,depthTest:true,depthWrite:false,transparent:true,opacity:.95});
  highlight=new THREE.Points(geom,mat);highlight.renderOrder=2;scene.add(highlight);
  points.material.opacity=.62;
  $('selection-info').textContent=`${c.point_count.toLocaleString()} anchored points · ${c.support_views} source views`;
  document.body.dataset.selectionCount=String(c.point_count);
  document.body.dataset.selectionFingerprint=c.point_ids.reduce((hash,id)=>Math.imul(hash^id,16777619)>>>0,2166136261).toString(16);
  [...$('candidates').children].forEach((b,i)=>b.classList.toggle('active',i===index));
  $('evidence-title').textContent=`Visual support for “${currentResult.query}”`;
  $('evidence-count').textContent='Detector regions · scores are not confidence';
  showPhotos(c.evidence);
  setView(c.evidence[0].frame_id);
  controls.target.set(...c.center);
  const offset=camera.position.clone().sub(controls.target);
  const radius=new THREE.Vector3(...c.bounds[1]).sub(new THREE.Vector3(...c.bounds[0])).length()/2;
  const minimumDistance=Math.max(.55,radius/Math.tan(THREE.MathUtils.degToRad(camera.fov/2))*1.25);
  if(offset.length()<minimumDistance)camera.position.copy(controls.target).add(offset.normalize().multiplyScalar(minimumDistance));
  camera.lookAt(controls.target);controls.update();
  $('view-label').textContent=`Focused region · source camera ${c.evidence[0].frame_id}`;
}

function present(result){
  currentResult=result;selectedCandidate=-1;clearHighlight();$('candidates').replaceChildren();
  const labels={match:'MULTI-VIEW MATCH',ambiguous:'NEEDS A CLOSER LOOK',no_evidence:'NO CONSISTENT EVIDENCE'};
  $('status-label').textContent=labels[result.status];
  $('result-title').textContent=result.status==='match'?result.query:result.status==='ambiguous'?'More than one possibility':'No supported match';
  $('result-message').textContent=result.message;
  result.candidates.forEach((c,i)=>{
    const b=document.createElement('button');b.className='candidate';
    const title=document.createElement('strong');title.textContent=`Region ${i+1} · ${c.support_views} source views`;
    const sub=document.createElement('span');sub.textContent=`${c.point_count.toLocaleString()} points · similarity ${c.similarity_score.toFixed(2)}`;
    b.append(title,sub);b.onclick=()=>selectCandidate(i);$('candidates').append(b);
  });
  if(result.candidates.length)selectCandidate(0);
  else{$('evidence-title').textContent='Nothing to verify yet';showPhotos([]);}
  document.body.dataset.queryStatus=result.status;
  document.body.dataset.queryLatency=String(result.latency_ms);
}

async function search(text){
  if(busy||!ready||!text.trim())return;
  const id=++requestId;busy=true;$('submit').disabled=true;
  $('status-label').textContent='CHECKING THE SCENE';$('result-title').textContent=text;
  $('result-message').textContent='Comparing the words to source views and checking consistent 3D support…';
  try{
    const response=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    const result=await response.json();if(!response.ok)throw Error(result.detail||'Search failed');
    if(id===requestId)present(result);
  }catch(e){if(id===requestId){$('status-label').textContent='SEARCH UNAVAILABLE';$('result-title').textContent='Please try again';$('result-message').textContent=e.message;}}
  finally{if(id===requestId){busy=false;$('submit').disabled=false;}}
}

$('search-form').onsubmit=e=>{e.preventDefault();search($('query').value);};
document.querySelectorAll('[data-query]').forEach(b=>b.onclick=()=>{$('query').value=b.dataset.query;search(b.dataset.query);});
$('home').onclick=()=>{if(meta)setView(meta.default_frame_id);};
$('rotate').onclick=()=>{if(!meta)return;const offset=camera.position.clone().sub(controls.target);offset.applyAxisAngle(camera.up.clone().normalize(),Math.PI/6);camera.position.copy(controls.target).add(offset);controls.update();$('view-label').textContent='Rotated viewpoint · anchored scene points';};
$('reset').onclick=()=>{if(!ready)return;requestId++;busy=false;$('submit').disabled=false;$('query').value='';currentResult=null;selectedCandidate=-1;clearHighlight();$('candidates').replaceChildren();$('status-label').textContent='READY TO EXPLORE';$('result-title').textContent='Start with something familiar';$('result-message').textContent='Choose an example above or enter your own object description.';$('evidence-title').textContent='The scene behind the reconstruction';$('evidence-count').textContent='Original RGB · 640 × 480';showPhotos([42,22,12,52].map(frame_id=>({frame_id})));setView(meta.default_frame_id);document.body.dataset.queryStatus='idle';document.body.dataset.selectionCount='0';document.body.dataset.selectionFingerprint='';};
$('close-photo').onclick=()=>$('photo-dialog').close();
$('photo-view').onclick=()=>{setView(activePhoto);$('photo-dialog').close();};
$('about-link').onclick=e=>{e.preventDefault();$('about-dialog').showModal();};
$('close-about').onclick=()=>$('about-dialog').close();

async function toggleNeural(){
  const button=$('neural-toggle');button.disabled=true;
  try{
    if(!neuralPoints){
      const response=await fetch('/api/learned-points');if(!response.ok)throw Error('Learned surface unavailable');
      const data=new Float32Array(await response.arrayBuffer()),n=data.length/6;
      const xyz=new Float32Array(n*3),rgb=new Float32Array(n*3);
      for(let i=0;i<n;i++){
        xyz.set(data.subarray(i*6,i*6+3),i*3);
        color.setRGB(data[i*6+3],data[i*6+4],data[i*6+5],THREE.SRGBColorSpace);rgb.set([color.r,color.g,color.b],i*3);
      }
      const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(xyz,3));geometry.setAttribute('color',new THREE.BufferAttribute(rgb,3));
      neuralPoints=new THREE.Points(geometry,new THREE.PointsMaterial({size:.014,vertexColors:true,depthWrite:true}));
      scene.add(neuralPoints);
    }
    neuralVisible=!neuralVisible;neuralPoints.visible=neuralVisible;points.visible=!neuralVisible;
    button.textContent=neuralVisible?'Show RGB-D surfels':'Show learned surface';
    $('point-info').textContent=neuralVisible?`${reconstructionMeta.surface_points.toLocaleString()} learned surface points · ${reconstructionMeta.steps.toLocaleString()} training steps`:`${meta.point_count.toLocaleString()} fitted points · ${meta.training_steps} optimization steps`;
    $('view-label').textContent=neuralVisible?'Learned surface proxy · original search anchors':'RGB-D surfels · original search anchors';
    document.body.dataset.reconstruction=neuralVisible?'learned':'surfel';
  }catch(e){$('result-message').textContent=e.message;}
  finally{button.disabled=false;}
}

function updateResearchImage(){
  const run=$('experiment-select').value,frame=Number($('frame-select').value);
  $('reconstruction-comparison').src=`/api/reconstruction/image/${run}/${frame}/comparison`;
  $('neural-original').src=`/api/photo/${frame}`;
  $('neural-render').src=`/api/reconstruction/image/${run}/${frame}/render`;
  const row=reconstructionMeta.runs[run].frames.find(x=>x.frame_id===frame);
  $('reconstruction-caption').textContent=`Frame ${frame} · PSNR ${row.psnr_db.toFixed(2)} dB · SSIM ${row.ssim.toFixed(3)} · depth MAE ${(row.depth_mae_m*100).toFixed(2)} cm. Error maps include occlusions, sensor noise and missing coverage; depth uses all valid reference pixels.`;
}

async function loadReconstruction(){
  try{
    const r=await fetch('/api/reconstruction').then(x=>x.json());if(!r.available)return;
    reconstructionMeta=r;
    $('neural-toggle').hidden=false;$('training-evidence').hidden=false;
    $('training-summary').textContent=`${r.parameter_count.toLocaleString()} learned parameters. ${r.steps.toLocaleString()} steps in the displayed checkpoint. ${r.train_views} training views, ${r.development_ids.length} development views and ${r.final_ids.length} final held-out views. Supplied camera poses; one real scene.`;
    $('research-acceptance').textContent=r.numerical_acceptance_pass?'The displayed production reconstruction passes the fixed numerical quality floors. The original custom-field study failed its color floors and remains below for inspection. These are repeated engineering checks, not an untouched benchmark.':'The displayed reconstruction did not pass all fixed quality floors. The failed results are preserved and labeled explicitly.';
    const metrics=[['Median view PSNR',`${r.runs[r.selected_run].summary.psnr_db.median.toFixed(2)} dB`],['Median view depth MAE',`${(r.runs[r.selected_run].summary.depth_mae_m.median*100).toFixed(2)} cm`],['Selected training run',`${(r.runs[r.selected_run].seconds/60).toFixed(1)} min`]];
    for(const [label,value] of metrics){const div=document.createElement('div'),strong=document.createElement('strong'),span=document.createElement('span');strong.textContent=value;span.textContent=label;div.append(strong,span);$('training-metrics').append(div);}
    const table=$('experiment-table');
    const header=document.createElement('tr');for(const label of ['Run','Train views','Steps','PSNR / dB','SSIM','Depth MAE / cm','Training / min']){const th=document.createElement('th');th.textContent=label;header.append(th);}table.append(header);
    for(const [name,run] of Object.entries(r.runs)){
      const option=document.createElement('option');option.value=name;option.textContent=run.label;$('experiment-select').append(option);
      const tr=document.createElement('tr');for(const value of [run.label,run.train_views||48,run.steps||6000,run.summary.psnr_db.median.toFixed(2),run.summary.ssim.median.toFixed(3),(run.summary.depth_mae_m.median*100).toFixed(2),(run.seconds/60).toFixed(1)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}table.append(tr);
    }
    $('experiment-select').value=r.selected_run;
    for(const id of r.final_ids){const option=document.createElement('option');option.value=id;option.textContent=`Frame ${id}`;$('frame-select').append(option);}
    $('frame-select').value=String(r.default_frame_id);
    $('research-limitations').textContent=r.limitations;
    // Curves show sampled training batches, explicitly distinct from held-out metrics.
    const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 900 190');svg.setAttribute('role','img');svg.setAttribute('aria-label','Training batch PSNR versus optimization step for all experiments');
    const colors=['#b7edca','#e6bb77','#9db6ef'];
    for(let db=5;db<=30;db+=5){const y=160-(db-5)/25*140,line=document.createElementNS(ns,'path');line.setAttribute('d',`M45 ${y}H880`);line.setAttribute('stroke','#35443b');svg.append(line);const t=document.createElementNS(ns,'text');t.setAttribute('x','4');t.setAttribute('y',String(y+4));t.textContent=`${db} dB`;svg.append(t);}
    (r.controlled_curve_runs||Object.keys(r.runs)).map(name=>r.runs[name]).forEach((run,i)=>{const path=document.createElementNS(ns,'path');path.setAttribute('d',run.history.map((p,j)=>`${j?'L':'M'}${45+p.step/(r.controlled_curve_steps||r.steps)*835} ${160-(p.batch_psnr-5)/25*140}`).join(' '));path.setAttribute('stroke',colors[i]);path.setAttribute('fill','none');path.setAttribute('stroke-width','2');svg.append(path);const t=document.createElementNS(ns,'text');t.setAttribute('x',String(55+i*270));t.setAttribute('y','17');t.setAttribute('fill',colors[i]);t.textContent=run.label;svg.append(t);});
    const label=document.createElementNS(ns,'text');label.setAttribute('x','300');label.setAttribute('y','186');label.textContent=`Optimization steps 0 – ${r.controlled_curve_steps||r.steps} · sampled training-batch PSNR`;svg.append(label);$('training-chart').append(svg);
    updateResearchImage();
    await toggleNeural();
  }catch(e){console.error('Reconstruction evidence unavailable',e);}
}
$('neural-toggle').onclick=toggleNeural;
$('training-evidence').onclick=()=>$('training-dialog').showModal();
$('close-training').onclick=()=>$('training-dialog').close();
$('experiment-select').onchange=updateResearchImage;$('frame-select').onchange=updateResearchImage;

async function start(){
  try{
    const [metadata,response]=await Promise.all([fetch('/api/scene').then(x=>{if(!x.ok)throw Error('Scene metadata unavailable');return x.json();}),fetch('/api/points')]);
    if(!response.ok)throw Error('Trained scene is unavailable');meta=metadata;
    const data=new Float32Array(await response.arrayBuffer());const n=data.length/6;
    positions=new Float32Array(n*3);baseColors=new Float32Array(n*3);
    for(let i=0;i<n;i++){
      positions.set(data.subarray(i*6,i*6+3),i*3);
      color.setRGB(data[i*6+3],data[i*6+4],data[i*6+5],THREE.SRGBColorSpace);baseColors.set([color.r,color.g,color.b],i*3);
    }
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.BufferAttribute(baseColors,3));
    points=new THREE.Points(geometry,new THREE.PointsMaterial({size:.012,vertexColors:true,transparent:true,opacity:1,depthWrite:true}));scene.add(points);
    resize();setView(meta.default_frame_id);$('loading').style.display='none';$('point-info').textContent=`${n.toLocaleString()} fitted points · ${meta.training_steps} optimization steps`;
    loadReconstruction();
    showPhotos([42,22,12,52].map(frame_id=>({frame_id})));
    for(let attempt=0;attempt<120;attempt++){
      const state=await fetch('/api/status').then(x=>x.json());
      if(state.error)throw Error(state.error);
      if(state.ready){ready=true;$('submit').disabled=false;document.querySelectorAll('[data-query]').forEach(button=>button.disabled=false);$('status-label').textContent='READY TO EXPLORE';$('result-title').textContent='Start with something familiar';$('result-message').textContent='Choose an example above or enter your own object description.';document.body.dataset.ready='true';return;}
      await new Promise(resolve=>setTimeout(resolve,500));
    }
    throw Error('The search index is taking longer than expected. Reload to retry.');
  }catch(e){$('status-label').textContent='SCENE UNAVAILABLE';$('result-title').textContent='Setup needs attention';$('result-message').textContent=e.message;$('loading').textContent='Scene could not be loaded';}
}
start();
