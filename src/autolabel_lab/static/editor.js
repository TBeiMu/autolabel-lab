'use strict';

// Image-space geometry stays independent of canvas scaling and device pixel ratio.
const WebAnnotator = (() => {
  const clone = value => JSON.parse(JSON.stringify(value));
  let E = null, loadGeneration = 0;
  const dialog = () => document.getElementById('annotation-dialog');
  const el = id => document.getElementById(id);
  function corners(box) {
    if (box.original) return box.original;
    const c=Math.cos(box.angle),s=Math.sin(box.angle);
    return [[-1,-1],[1,-1],[1,1],[-1,1]].map(([x,y])=>[box.cx+x*box.w/2*c-y*box.h/2*s,box.cy+x*box.w/2*s+y*box.h/2*c]);
  }
  function fromShape(shape) {
    let pts=clone(shape.points);
    if(pts.length===2){const [a,b]=pts;pts=[a,[b[0],a[1]],b,[a[0],b[1]]]}
    return {label:shape.label,score:shape.score,cx:pts.reduce((n,p)=>n+p[0],0)/4,cy:pts.reduce((n,p)=>n+p[1],0)/4,
      w:Math.hypot(pts[1][0]-pts[0][0],pts[1][1]-pts[0][1]),h:Math.hypot(pts[2][0]-pts[1][0],pts[2][1]-pts[1][1]),
      angle:Math.atan2(pts[1][1]-pts[0][1],pts[1][0]-pts[0][0]),original:pts};
  }
  function local(box,p) {const dx=p[0]-box.cx,dy=p[1]-box.cy,c=Math.cos(box.angle),s=Math.sin(box.angle);return [dx*c+dy*s,-dx*s+dy*c]}
  function color(box) {return COLORS[Math.max(0,E.classes.indexOf(box.label))%COLORS.length]}
  function key(frame=E.frame) {return `alab-annotation:${E.projectId}:${E.round||E.subset}:${frame.id}`}
  function draft() {
    if(!E||E.readOnly||!E.frame)return;
    try { if(E.dirty)localStorage.setItem(key(),JSON.stringify({boxes:E.boxes,revision:E.revision}));else localStorage.removeItem(key()); }
    catch { message('浏览器无法暂存草稿，请及时点击保存。',true); }
  }
  function message(text,error=false) {if(el('ann-status')){el('ann-status').textContent=text;el('ann-status').classList.toggle('danger-text',error)}}
  function snapshot() {E.undo.push(clone(E.boxes));if(E.undo.length>80)E.undo.shift();E.redo=[]}
  function changed() {E.dirty=true;draft();renderList();draw();message('有未保存修改 · Ctrl+S 保存')}
  function available() {return E&&!E.busy&&!E.readOnly}
  function setBusy(busy) {E.busy=busy;renderList()}
  function params(frame) {return `id=${encodeURIComponent(E.projectId)}&frame=${encodeURIComponent(frame.id)}&${E.round?'round='+encodeURIComponent(E.round):'subset='+E.subset}`}
  async function fetchAnnotation(frame) {return api('/api/annotation?'+params(frame))}
  function mount() {
    dialog().innerHTML=`<div class="ann-header"><div><div class="eyebrow">WEB ANNOTATION / OBB</div><h2 id="ann-title">正在加载图片</h2><span id="ann-file"></span></div><button class="btn" id="ann-close">保存并关闭</button></div>
      <div class="ann-toolbar"><button class="btn" id="ann-select">选择 / 调整 V</button><button class="btn" id="ann-draw">画框 D</button>
      <button class="btn" id="ann-undo">撤销</button><button class="btn" id="ann-redo">重做</button><button class="btn" id="ann-delete">删除框</button>
      <button class="btn" id="ann-copy">复制上一张框 C</button><button class="btn" id="ann-zoom-out">−</button><span id="ann-zoom"></span><button class="btn" id="ann-zoom-in">＋</button><button class="btn" id="ann-fit">适应窗口</button>
      ${E.round?'<label class="ann-toggle"><input id="ann-show-raw" type="checkbox" checked>对照原始预测</label>':''}</div>
      <div class="ann-layout"><div class="ann-images"><div class="ann-stage" id="ann-stage"><canvas id="ann-canvas" tabindex="0" aria-label="图片标注画布"></canvas></div>${E.round?'<div class="ann-raw" id="ann-raw-panel"><h3>原始预测 · 只读</h3><canvas id="ann-raw"></canvas></div>':''}</div>
      <aside class="ann-inspector"><label for="ann-class">框的类别 / 新框类别</label><select id="ann-class">${E.classes.map(c=>`<option value="${esc(c)}">${esc(c)}</option>`).join('')}</select><div class="field"><label for="ann-angle">旋转角度（度）</label><input id="ann-angle" type="number" step="1"></div>
      <div class="ann-boxes" id="ann-boxes"></div><p class="form-help">拖动画框；选择框后拖动内部移动、四角调整尺寸、圆点旋转。滚轮缩放，按空格拖动平移。框须在图片边界内。</p><p class="form-help">复制上一张会追加框，之后请按当前图片调整。无目标图片也要保存；审核时保存表示已检查此张。</p>
      <button class="btn" id="ann-reload">丢弃草稿并重新加载</button></aside></div>
      <div class="ann-footer"><div><strong id="ann-position"></strong><span id="ann-status" role="status"></span></div><div class="buttons"><button class="btn" id="ann-prev">← 上一张</button><button class="btn primary" id="ann-save">保存 Ctrl+S</button><button class="btn primary" id="ann-next">保存并下一张 →</button></div></div>`;
    el('ann-close').onclick=()=>action(close);
    el('ann-save').onclick=()=>action(save);
    el('ann-prev').onclick=()=>action(()=>navigate(-1));el('ann-next').onclick=()=>action(()=>navigate(1));
    el('ann-select').onclick=()=>mode('select');el('ann-draw').onclick=()=>mode('draw');
    el('ann-copy').onclick=()=>action(copyPrevious);el('ann-delete').onclick=remove;
    el('ann-undo').onclick=()=>history('undo');el('ann-redo').onclick=()=>history('redo');
    el('ann-fit').onclick=fit;el('ann-zoom-in').onclick=()=>zoom(1.25);el('ann-zoom-out').onclick=()=>zoom(.8);
    el('ann-class').onchange=()=>{if(!available())return;E.label=el('ann-class').value;if(E.selected>=0){snapshot();E.boxes[E.selected].label=E.label;changed()}};
    el('ann-angle').onchange=()=>{if(!available()||E.selected<0)return;const angle=Number(el('ann-angle').value);if(!Number.isFinite(angle))return;snapshot();const b=E.boxes[E.selected];b.angle=angle*Math.PI/180;delete b.original;changed()};
    el('ann-reload').onclick=()=>action(async()=>{if(!confirm('丢弃本张未保存的草稿并读取磁盘标注？'))return;localStorage.removeItem(key());await load(E.index,false)});
    if(el('ann-show-raw'))el('ann-show-raw').onchange=()=>{el('ann-raw-panel').hidden=!el('ann-show-raw').checked;requestAnimationFrame(fit)};
    const canvas=el('ann-canvas');canvas.onpointerdown=pointerDown;canvas.onpointermove=pointerMove;canvas.onpointerup=pointerUp;canvas.onpointercancel=()=>{if(E?.gesture){E.boxes=E.gesture.before;E.gesture=null;renderList();draw()}};
    canvas.onwheel=event=>{event.preventDefault();zoom(event.deltaY<0?1.12:1/1.12,[event.offsetX,event.offsetY])};
    E.observer=new ResizeObserver(()=>{if(E?.image)fit()});E.observer.observe(el('ann-stage'));
  }
  async function action(fn) {try {await fn()}catch(error){message(error.message,true);toast(error.message,true)}}
  async function open(frameId,roundName='',subset='') {
    if(E&&dialog().open){await close();if(dialog().open)return}
    const frames=roundName?S.details.manifest.frames.filter(f=>S.details.rounds.find(r=>r.name===roundName)?.frame_ids.includes(f.id)):
      S.details.manifest.frames.filter(f=>f.role===(subset||S.details.manifest.frames.find(f=>f.id===frameId)?.role));
    if(!frames.length)throw new Error('此分组暂无图片。');
    const index=Math.max(0,frames.findIndex(f=>f.id===frameId));
    E={projectId:S.id,classes:[...S.details.config.classes],frames,round:roundName,subset:subset||frames[index].role,index,
      label:S.details.config.classes[0],boxes:[],selected:-1,undo:[],redo:[],dirty:false,busy:true,mode:'select',space:false};
    mount();dialog().showModal();await load(index);el('ann-canvas').focus();
  }
  async function load(index,restore=true) {
    const generation=++loadGeneration;setBusy(true);E.index=index;E.frame=E.frames[index];E.image=null;
    message('正在加载图片与标注…');
    try {
      const data=await fetchAnnotation(E.frame),image=new Image();image.src=`/api/file?id=${encodeURIComponent(E.projectId)}&path=${encodeURIComponent('frames/'+E.frame.file)}`;
      await image.decode();if(generation!==loadGeneration||!E)return;
      E.image=image;E.readOnly=data.read_only;E.revision=data.revision;E.boxes=data.document.shapes.map(fromShape);E.raw=data.raw;
      E.dirty=false;E.undo=[];E.redo=[];E.selected=-1;E.gesture=null;
      let restored=false;
      if(restore&&!E.readOnly){try{const saved=JSON.parse(localStorage.getItem(key()));if(saved){E.boxes=saved.boxes;E.revision=saved.revision;E.dirty=true;restored=true}}catch{}}
      el('ann-title').textContent=(E.round?'审核 '+E.round:ROLE[E.subset])+' · '+frameCaption(E.frame)+(E.readOnly?' · 已归档，只读':'');
      el('ann-file').textContent=(E.frame.source_name||E.frame.file)+` · ${E.frame.width} × ${E.frame.height}`;
      el('ann-position').textContent=`${index+1} / ${E.frames.length}`;
      if(E.raw)drawBoxes(el('ann-raw'),image,E.raw);
      setBusy(false);fit();mode('select');
      message(restored?'已恢复未保存草稿；保存前会核验磁盘版本。':E.readOnly?'此标注已固定归档，只能查看。':data.reviewed?'此张已保存并确认，可继续调整。':data.saved?'已加载现有标注；请检查后保存。':'尚未标注，画框或保存空标注。');
    }catch(error){if(E&&generation===loadGeneration){setBusy(false);E.readOnly=true;E.image=null;renderList();message('加载失败，请重试：'+error.message,true)}throw error}
  }
  function size() {const canvas=el('ann-canvas'),rect=el('ann-stage').getBoundingClientRect();const dpr=devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));E.dpr=dpr;E.viewport=[rect.width,rect.height]}
  function fit() {if(!E?.image)return;size();E.scale=Math.min(E.viewport[0]/E.frame.width,E.viewport[1]/E.frame.height)*.95;E.offset=[(E.viewport[0]-E.frame.width*E.scale)/2,(E.viewport[1]-E.frame.height*E.scale)/2];draw()}
  function zoom(factor,point) {if(!E?.image)return;point=point||E.viewport.map(v=>v/2);const scale=Math.max(.01,Math.min(20,E.scale*factor));E.offset=point.map((v,i)=>v-(v-E.offset[i])*scale/E.scale);E.scale=scale;draw()}
  function canvasPoint(event) {const rect=el('ann-canvas').getBoundingClientRect();return [event.clientX-rect.left,event.clientY-rect.top]}
  function imagePoint(event) {return canvasPoint(event).map((v,i)=>(v-E.offset[i])/E.scale)}
  function handle(box) {return [box.cx+Math.sin(box.angle)*(box.h/2+28/E.scale),box.cy-Math.cos(box.angle)*(box.h/2+28/E.scale)]}
  function draw() {
    if(!E?.image)return;const ctx=el('ann-canvas').getContext('2d');ctx.setTransform(E.dpr,0,0,E.dpr,0,0);ctx.fillStyle='#152421';ctx.fillRect(0,0,...E.viewport);
    ctx.translate(...E.offset);ctx.scale(E.scale,E.scale);ctx.drawImage(E.image,0,0,E.frame.width,E.frame.height);
    E.boxes.forEach((box,index)=>{const pts=corners(box),selected=index===E.selected;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.strokeStyle=color(box);ctx.lineWidth=(selected?3:1.8)/E.scale;ctx.stroke();
      if(selected){ctx.fillStyle=color(box);ctx.globalAlpha=.13;ctx.fill();ctx.globalAlpha=1}
      ctx.font=`${12/E.scale}px Segoe UI,Microsoft YaHei`;ctx.fillStyle=color(box);const text=`${index+1}. ${box.label}`;const x=Math.max(0,pts[0][0]),y=Math.max(16/E.scale,pts[0][1]);ctx.fillRect(x,y-17/E.scale,ctx.measureText(text).width+8/E.scale,19/E.scale);ctx.fillStyle='white';ctx.fillText(text,x+4/E.scale,y-3/E.scale);
      if(selected&&!E.readOnly){ctx.fillStyle='white';for(const p of pts){ctx.fillRect(p[0]-5/E.scale,p[1]-5/E.scale,10/E.scale,10/E.scale);ctx.strokeRect(p[0]-5/E.scale,p[1]-5/E.scale,10/E.scale,10/E.scale)}const hp=handle(box);ctx.beginPath();ctx.moveTo(box.cx+Math.sin(box.angle)*box.h/2,box.cy-Math.cos(box.angle)*box.h/2);ctx.lineTo(...hp);ctx.stroke();ctx.beginPath();ctx.arc(...hp,6/E.scale,0,Math.PI*2);ctx.fill();ctx.stroke()}
    });
    if(E.gesture?.type==='draw'){const a=E.gesture.start,b=E.gesture.current;ctx.strokeStyle='white';ctx.lineWidth=2/E.scale;ctx.setLineDash([6/E.scale,4/E.scale]);ctx.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);ctx.setLineDash([])}
    el('ann-zoom').textContent=Math.round(E.scale*100)+'%';
  }
  function renderList() {
    if(!E)return;el('ann-boxes').innerHTML=E.boxes.map((b,i)=>`<button class="ann-box ${i===E.selected?'selected':''}" data-box="${i}" style="border-left-color:${color(b)}"><strong>${i+1}. ${esc(b.label)}</strong><small>${Math.round(b.w)} × ${Math.round(b.h)} · ${(b.angle*180/Math.PI).toFixed(1)}°${b.score!=null?' · 预测 '+Number(b.score).toFixed(2):''}</small></button>`).join('')||'<p>当前没有框</p>';
    el('ann-boxes').querySelectorAll('[data-box]').forEach(b=>b.onclick=()=>{if(E.busy)return;E.selected=Number(b.dataset.box);renderList();draw()});
    for(const id of ['ann-select','ann-draw','ann-class','ann-copy','ann-save'])el(id).disabled=E.busy||E.readOnly;
    el('ann-copy').disabled||=E.index===0;
    for(const id of ['ann-delete','ann-angle'])el(id).disabled=E.busy||E.readOnly||E.selected<0;
    el('ann-undo').disabled=E.busy||E.readOnly||!E.undo.length;el('ann-redo').disabled=E.busy||E.readOnly||!E.redo.length;
    el('ann-prev').disabled=E.busy||E.index===0;el('ann-next').disabled=E.busy||E.index===E.frames.length-1;el('ann-close').disabled=E.busy;
    el('ann-reload').disabled=E.busy;el('ann-next').textContent=E.readOnly?'下一张 →':'保存并下一张 →';el('ann-close').textContent=E.readOnly?'关闭':'保存并关闭';
    if(E.selected>=0){const b=E.boxes[E.selected];el('ann-class').value=b.label;el('ann-angle').value=(b.angle*180/Math.PI).toFixed(1)}else{el('ann-class').value=E.label;el('ann-angle').value=''}
  }
  function mode(value) {if(!E)return;E.mode=value;el('ann-draw').classList.toggle('primary',value==='draw');el('ann-select').classList.toggle('primary',value==='select');el('ann-canvas').style.cursor=value==='draw'?'crosshair':'default'}
  function pointerDown(event) {
    if(!E?.image||E.busy)return;const p=imagePoint(event);el('ann-canvas').focus();
    if(E.space||event.button===1){event.preventDefault();E.gesture={type:'pan',start:canvasPoint(event),offset:[...E.offset],before:clone(E.boxes)};el('ann-canvas').setPointerCapture(event.pointerId);return}
    if(!available()||event.button!==0)return;
    const before=clone(E.boxes);let gesture;
    if(E.mode==='draw'){const start=p.map((v,i)=>Math.max(0,Math.min(i?E.frame.height:E.frame.width,v)));gesture={type:'draw',start,current:start,before}}
    else {
      const selected=E.boxes[E.selected];
      if(selected){const pts=corners(selected),corner=pts.findIndex(q=>Math.hypot(q[0]-p[0],q[1]-p[1])*E.scale<10);const hp=handle(selected);
        if(Math.hypot(hp[0]-p[0],hp[1]-p[1])*E.scale<12)gesture={type:'rotate',box:clone(selected),before,start:p};
        else if(corner>=0)gesture={type:'resize',box:clone(selected),before,opposite:pts[(corner+2)%4],start:p};}
      if(!gesture){E.selected=-1;for(let i=E.boxes.length-1;i>=0;i--){const b=E.boxes[i],q=local(b,p);if(Math.abs(q[0])<=b.w/2&&Math.abs(q[1])<=b.h/2){E.selected=i;break}}if(E.selected>=0)gesture={type:'move',box:clone(E.boxes[E.selected]),start:p,before}}
    }
    E.gesture=gesture;renderList();draw();if(gesture)el('ann-canvas').setPointerCapture(event.pointerId);
  }
  function pointerMove(event) {
    if(!E?.gesture)return;const g=E.gesture,p=imagePoint(event);
    if(g.type==='pan'){const pos=canvasPoint(event);E.offset=g.offset.map((v,i)=>v+pos[i]-g.start[i]);draw();return}
    if(g.type==='draw')g.current=p.map((v,i)=>Math.max(0,Math.min(i?E.frame.height:E.frame.width,v)));
    else {const b=E.boxes[E.selected];if(!b)return;
      if(g.type==='move'){const pts=corners(g.box),dx=Math.max(-Math.min(...pts.map(q=>q[0])),Math.min(E.frame.width-Math.max(...pts.map(q=>q[0])),p[0]-g.start[0])),dy=Math.max(-Math.min(...pts.map(q=>q[1])),Math.min(E.frame.height-Math.max(...pts.map(q=>q[1])),p[1]-g.start[1]));b.cx=g.box.cx+dx;b.cy=g.box.cy+dy;if(g.box.original)b.original=g.box.original.map(q=>[q[0]+dx,q[1]+dy])}
      if(g.type==='rotate'){b.angle=Math.atan2(p[1]-b.cy,p[0]-b.cx)+Math.PI/2;if(event.shiftKey)b.angle=Math.round(b.angle/(Math.PI/12))*Math.PI/12;delete b.original}
      if(g.type==='resize'){const dx=p[0]-g.opposite[0],dy=p[1]-g.opposite[1],c=Math.cos(g.box.angle),s=Math.sin(g.box.angle);b.w=Math.max(1,Math.abs(dx*c+dy*s));b.h=Math.max(1,Math.abs(-dx*s+dy*c));b.cx=(p[0]+g.opposite[0])/2;b.cy=(p[1]+g.opposite[1])/2;delete b.original}
    }draw();
  }
  function pointerUp(event) {
    if(!E?.gesture)return;const g=E.gesture;E.gesture=null;
    if(el('ann-canvas').hasPointerCapture(event.pointerId))el('ann-canvas').releasePointerCapture(event.pointerId);
    if(g.type==='pan')return;
    if(g.type==='draw'){const [a,b]=[g.start,g.current],w=Math.abs(b[0]-a[0]),h=Math.abs(b[1]-a[1]);if(w<1||h<1){draw();return}E.boxes.push({label:E.label,cx:(a[0]+b[0])/2,cy:(a[1]+b[1])/2,w,h,angle:0});E.selected=E.boxes.length-1}
    if(JSON.stringify(g.before)!==JSON.stringify(E.boxes)){E.undo.push(g.before);if(E.undo.length>80)E.undo.shift();E.redo=[];changed()}else{renderList();draw()}
  }
  function remove() {if(!available()||E.selected<0)return;snapshot();E.boxes.splice(E.selected,1);E.selected=-1;changed()}
  function history(direction) {if(!available()||!E[direction].length)return;const other=direction==='undo'?'redo':'undo';E[other].push(clone(E.boxes));E.boxes=E[direction].pop();E.selected=-1;changed()}
  async function copyPrevious() {
    if(!available()||E.index===0)return;setBusy(true);
    try{const prev=E.frames[E.index-1],data=await fetchAnnotation(prev);if(!data.saved)throw new Error('上一张还没有保存标注。');const shapes=data.document.shapes.map(shape=>({...shape,points:shape.points.map(p=>[p[0]*E.frame.width/prev.width,p[1]*E.frame.height/prev.height])}));
      if(!shapes.length)throw new Error('上一张没有框可复制。');snapshot();E.boxes.push(...shapes.map(fromShape));E.selected=E.boxes.length-1;changed();message(`已追加上一张的 ${shapes.length} 个框，请根据当前图片调整并保存。`);
    }finally{setBusy(false)}
  }
  async function save() {
    if(!E||E.readOnly||E.busy)return false;if(E.gesture)throw new Error('请先完成当前框选操作。');
    const shapes=E.boxes.map(box=>({label:box.label,points:corners(box),shape_type:'rotation',flags:{}}));
    if(shapes.some(shape=>shape.points.some(p=>p[0]<0||p[1]<0||p[0]>E.frame.width||p[1]>E.frame.height)))throw new Error('有框超出图片边界，请调整后保存。');
    setBusy(true);
    try{const result=await api('/api/annotation',{project_id:E.projectId,frame:E.frame.id,subset:E.round?undefined:E.subset,round:E.round||undefined,revision:E.revision,shapes});E.revision=result.revision;E.dirty=false;E.boxes=result.document.shapes.map(fromShape);draft();message(E.round?'已保存并确认此张审核。':'已保存标注。');return true}
    catch(error){draft();throw error}finally{setBusy(false);draw()}
  }
  async function navigate(delta) {if(!E||E.busy)return;const index=E.index+delta;if(index<0||index>=E.frames.length)return;if(!E.readOnly&&E.image)await save();await load(index)}
  async function close() {if(!E||E.busy)return;if(!E.readOnly&&E.image)await save();E.observer?.disconnect();E=null;++loadGeneration;dialog().close();await refresh(true)}
  dialog().addEventListener('cancel',event=>{event.preventDefault();action(close)});
  window.addEventListener('beforeunload',event=>{if(E?.dirty){draft();event.preventDefault();event.returnValue=''}});
  document.addEventListener('keydown',event=>{
    if(!E||!dialog().open)return;
    if(event.code==='Space'&&!/INPUT|SELECT|TEXTAREA/.test(event.target.tagName)){event.preventDefault();E.space=true;return}
    if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='s'){event.preventDefault();action(save);return}
    if(/INPUT|SELECT|TEXTAREA/.test(event.target.tagName))return;
    if(E.busy)return;
    const key=event.key.toLowerCase();
    if((event.ctrlKey||event.metaKey)&&['z','y'].includes(key)){event.preventDefault();history(key==='y'||event.shiftKey?'redo':'undo')}
    else if(key==='d')mode('draw');else if(key==='v')mode('select');else if(key==='c')action(copyPrevious);
    else if(event.key==='Delete'||event.key==='Backspace'){event.preventDefault();remove()}
    else if(event.key==='ArrowRight'){event.preventDefault();action(()=>navigate(1))}else if(event.key==='ArrowLeft'){event.preventDefault();action(()=>navigate(-1))}
  });
  document.addEventListener('keyup',event=>{if(E&&event.code==='Space')E.space=false});
  window.addEventListener('blur',()=>{if(E)E.space=false});
  return {open, corners};
})();
