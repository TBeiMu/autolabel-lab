'use strict';

// Image-space geometry stays independent of canvas scaling and device pixel ratio.
const WebAnnotator = (() => {
  const clone = value => JSON.parse(JSON.stringify(value));
  let E = null, loadGeneration = 0, clipboard = null, boxSequence = 0;
  const boxId = () => 'box-'+(++boxSequence);
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
    return {id:boxId(),label:shape.label,score:shape.score,cx:pts.reduce((n,p)=>n+p[0],0)/4,cy:pts.reduce((n,p)=>n+p[1],0)/4,
      w:Math.hypot(pts[1][0]-pts[0][0],pts[1][1]-pts[0][1]),h:Math.hypot(pts[2][0]-pts[1][0],pts[2][1]-pts[1][1]),
      angle:Math.atan2(pts[1][1]-pts[0][1],pts[1][0]-pts[0][0]),original:pts};
  }
  function local(box,p) {const dx=p[0]-box.cx,dy=p[1]-box.cy,c=Math.cos(box.angle),s=Math.sin(box.angle);return [dx*c+dy*s,-dx*s+dy*c]}
  function color(box) {return COLORS[Math.max(0,E.classes.indexOf(box.label))%COLORS.length]}
  function outside(box) {return corners(box).some(p=>!Number.isFinite(p[0])||!Number.isFinite(p[1])||p[0]<0||p[1]<0||p[0]>E.frame.width||p[1]>E.frame.height)}
  function boxRows() {return E.boxes.map((box,index)=>({box,index,number:index+1,invalid:outside(box)}))}
  function selection(ids,primary=ids.at(-1)) {E.selection=new Set(ids);E.selected=E.boxes.findIndex(b=>b.id===primary)}
  function selectedBoxes() {return E.boxes.filter(box=>E.selection.has(box.id))}
  function selectBox(id,add=false) {
    if(!E||E.busy||E.gesture||E.orderGesture)return;
    if(add){const ids=new Set(E.selection);ids.has(id)?ids.delete(id):ids.add(id);selection([...ids],ids.has(id)?id:[...ids].at(-1))}
    else selection([id],id);
    renderList();draw();
  }
  function checkBounds() {
    if(!E?.frame||!el('ann-bounds'))return;
    const rows=boxRows(),byId=new Map(rows.map(row=>[row.box.id,row])),bad=rows.filter(row=>row.invalid);el('ann-bounds').hidden=!bad.length;
    el('ann-bounds').innerHTML=bad.length?`<strong>⚠ ${bad.length} 个框超出图片边界</strong><span>点击序号定位，调整后才能保存。</span><div>${bad.map(row=>`<button class="ann-check-box" data-check-box="${row.box.id}">检查框 ${row.number}</button>`).join('')}</div>`:'';
    el('ann-bounds').querySelectorAll('[data-check-box]').forEach(b=>b.onclick=()=>focusBox(b.dataset.checkBox));
    el('ann-boxes').querySelectorAll('[data-box-id]').forEach(button=>{const invalid=byId.get(button.dataset.boxId)?.invalid;button.classList.toggle('out-of-bounds',!!invalid);button.closest('.ann-box-row').classList.toggle('out-of-bounds',!!invalid);const flag=button.querySelector('.ann-bound-flag');if(flag)flag.hidden=!invalid});
  }
  function focusBox(id) {
    if(!E?.image||E.busy||E.gesture)return;selectBox(id);if(E.selected<0)return;
    const pts=corners(E.boxes[E.selected]),xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
    const left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys);
    E.scale=Math.max(.01,Math.min(4,(E.viewport[0]-60)/Math.max(1,right-left),(E.viewport[1]-60)/Math.max(1,bottom-top)));
    E.offset=[E.viewport[0]/2-(left+right)/2*E.scale,E.viewport[1]/2-(top+bottom)/2*E.scale];draw();
    el('ann-boxes').querySelector(`[data-box-id="${id}"]`)?.scrollIntoView({block:'nearest'});el('ann-canvas').focus();
  }
  function key(frame=E.frame) {return `alab-annotation:${E.projectId}:${E.round||E.subset}:${frame.id}`}
  function draft() {
    if(!E||E.readOnly||!E.frame)return;
    try { if(E.dirty)localStorage.setItem(key(),JSON.stringify({boxes:E.boxes,revision:E.revision}));else localStorage.removeItem(key()); }
    catch { message('浏览器无法暂存草稿，请及时点击保存。',true); }
  }
  function message(text,error=false) {if(el('ann-status')){el('ann-status').textContent=text;el('ann-status').classList.toggle('danger-text',error)}}
  function snapshot() {E.undo.push(clone(E.boxes));if(E.undo.length>80)E.undo.shift();E.redo=[]}
  function changed() {E.dirty=true;draft();renderList();draw();message('有未保存修改 · Ctrl+S 保存')}
  function available() {return E&&!E.busy&&!E.readOnly&&!E.gesture&&!E.orderGesture}
  function setBusy(busy) {E.busy=busy;renderList()}
  function params(frame) {return `id=${encodeURIComponent(E.projectId)}&frame=${encodeURIComponent(frame.id)}&${E.round?'round='+encodeURIComponent(E.round):'subset='+E.subset}`}
  async function fetchAnnotation(frame) {return api('/api/annotation?'+params(frame))}
  function mount() {
    dialog().innerHTML=`<div class="ann-header"><div><div class="eyebrow">WEB ANNOTATION / OBB</div><h2 id="ann-title">正在加载图片</h2><span id="ann-file"></span></div><button class="btn" id="ann-close">保存并关闭</button></div>
      <div class="ann-toolbar"><button class="btn" id="ann-select">选择 / 调整 V</button><button class="btn" id="ann-marquee">拖选多个框 M</button><button class="btn" id="ann-draw">画框 D</button><span id="ann-selection-count"></span>
      <button class="btn" id="ann-undo">撤销</button><button class="btn" id="ann-redo">重做</button><button class="btn" id="ann-delete">删除框</button>
      <button class="btn" id="ann-copy-box">复制框 Ctrl+C</button><button class="btn" id="ann-paste-box">粘贴框 Ctrl+V</button>
      <button class="btn" id="ann-copy">复制上一张框 C</button><button class="btn" id="ann-zoom-out">−</button><span id="ann-zoom"></span><button class="btn" id="ann-zoom-in">＋</button><button class="btn" id="ann-fit">适应窗口</button>
      ${E.round?'<label class="ann-toggle"><input id="ann-show-raw" type="checkbox" checked>对照原始预测</label>':''}</div>
      <div class="ann-layout"><div class="ann-images"><div class="ann-stage" id="ann-stage"><canvas id="ann-canvas" tabindex="0" aria-label="图片标注画布"></canvas></div>${E.round?'<div class="ann-raw" id="ann-raw-panel"><h3>原始预测 · 只读</h3><canvas id="ann-raw"></canvas></div>':''}</div>
      <aside class="ann-inspector"><label for="ann-class">框的类别 / 新框类别</label><select id="ann-class">${E.classes.map(c=>`<option value="${esc(c)}">${esc(c)}</option>`).join('')}</select><div class="field"><label for="ann-angle">旋转角度（度）</label><input id="ann-angle" type="number" step="1"></div>
      <div id="ann-bounds" class="ann-bounds" role="alert" hidden></div><div class="ann-boxes" id="ann-boxes"></div><p class="form-help">M 拖选多个框，V 选择后一起移动；Shift 点击增减选择。单个框可拖动四角和圆点调整。右侧拖动 ⠿ 调整序号，选中多个框时一起排序。滚轮缩放，空格拖动平移。</p><p class="form-help">选中框后 Ctrl+C 复制、Ctrl+V 粘贴；可粘到其他图片。C 复制上一张全部框。无目标图片也要保存；审核时保存表示已检查此张。</p>
      <button class="btn" id="ann-reload">丢弃草稿并重新加载</button></aside></div>
      <div class="ann-footer"><div><strong id="ann-position"></strong><span id="ann-status" role="status"></span></div><div class="buttons"><button class="btn" id="ann-prev">← 上一张</button><button class="btn primary" id="ann-save">保存 Ctrl+S</button><button class="btn primary" id="ann-next">保存并下一张 →</button></div></div>`;
    el('ann-close').onclick=()=>action(close);
    el('ann-save').onclick=()=>action(save);
    el('ann-prev').onclick=()=>action(()=>navigate(-1));el('ann-next').onclick=()=>action(()=>navigate(1));
    el('ann-select').onclick=()=>mode('select');el('ann-draw').onclick=()=>mode('draw');
    el('ann-marquee').onclick=()=>mode('marquee');
    el('ann-copy').onclick=()=>action(copyPrevious);el('ann-delete').onclick=remove;
    el('ann-copy-box').onclick=copyBox;el('ann-paste-box').onclick=()=>action(pasteBox);
    el('ann-undo').onclick=()=>history('undo');el('ann-redo').onclick=()=>history('redo');
    el('ann-fit').onclick=fit;el('ann-zoom-in').onclick=()=>zoom(1.25);el('ann-zoom-out').onclick=()=>zoom(.8);
    el('ann-class').onchange=()=>{if(!available())return;E.label=el('ann-class').value;if(E.selection.size){snapshot();selectedBoxes().forEach(box=>box.label=E.label);changed()}};
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
      label:S.details.config.classes[0],boxes:[],selected:-1,selection:new Set(),undo:[],redo:[],dirty:false,busy:true,mode:'select',space:false};
    mount();dialog().showModal();await load(index);el('ann-canvas').focus();
  }
  async function load(index,restore=true) {
    const generation=++loadGeneration;setBusy(true);E.index=index;E.frame=E.frames[index];E.image=null;
    message('正在加载图片与标注…');
    try {
      const data=await fetchAnnotation(E.frame),image=new Image();image.src=`/api/file?id=${encodeURIComponent(E.projectId)}&path=${encodeURIComponent('frames/'+E.frame.file)}`;
      await image.decode();if(generation!==loadGeneration||!E)return;
      E.image=image;E.readOnly=data.read_only;E.revision=data.revision;E.boxes=data.document.shapes.map(fromShape);E.raw=data.raw;
      E.dirty=false;E.undo=[];E.redo=[];selection([]);E.gesture=null;E.orderGesture=null;
      let restored=false;
      if(restore&&!E.readOnly){try{const saved=JSON.parse(localStorage.getItem(key()));if(saved){E.boxes=saved.boxes;E.revision=saved.revision;E.dirty=true;restored=true}}catch{}}
      E.boxes.forEach(box=>box.id=boxId());
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
    boxRows().forEach(({box,index,number,invalid})=>{const pts=corners(box),selected=E.selection.has(box.id),stroke=invalid?'#dc3f36':color(box);ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.strokeStyle=stroke;ctx.lineWidth=(invalid?4:selected?3:1.8)/E.scale;ctx.setLineDash(invalid?[7/E.scale,4/E.scale]:[]);ctx.stroke();ctx.setLineDash([]);
      if(selected){ctx.fillStyle=color(box);ctx.globalAlpha=.13;ctx.fill();ctx.globalAlpha=1}
      ctx.font=`${12/E.scale}px Segoe UI,Microsoft YaHei`;ctx.fillStyle=stroke;const text=`${number}. ${box.label}${invalid?' · 越界':''}`;const x=Math.max(0,Math.min(pts[0][0],E.frame.width-(ctx.measureText(text).width+8/E.scale))),y=Math.max(16/E.scale,Math.min(pts[0][1],E.frame.height-2/E.scale));ctx.fillRect(x,y-17/E.scale,ctx.measureText(text).width+8/E.scale,19/E.scale);ctx.fillStyle='white';ctx.fillText(text,x+4/E.scale,y-3/E.scale);
      if(selected&&E.selection.size===1&&!E.readOnly){ctx.fillStyle='white';for(const p of pts){ctx.fillRect(p[0]-5/E.scale,p[1]-5/E.scale,10/E.scale,10/E.scale);ctx.strokeRect(p[0]-5/E.scale,p[1]-5/E.scale,10/E.scale,10/E.scale)}const hp=handle(box);ctx.beginPath();ctx.moveTo(box.cx+Math.sin(box.angle)*box.h/2,box.cy-Math.cos(box.angle)*box.h/2);ctx.lineTo(...hp);ctx.stroke();ctx.beginPath();ctx.arc(...hp,6/E.scale,0,Math.PI*2);ctx.fill();ctx.stroke()}
    });
    if(['draw','marquee'].includes(E.gesture?.type)){const a=E.gesture.start,b=E.gesture.current;ctx.strokeStyle=E.gesture.type==='marquee'?'#79dfc1':'white';ctx.lineWidth=2/E.scale;ctx.setLineDash([6/E.scale,4/E.scale]);ctx.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);ctx.setLineDash([])}
    el('ann-zoom').textContent=Math.round(E.scale*100)+'%';
    checkBounds();
  }
  function renderList() {
    if(!E)return;if(E.selected>=E.boxes.length)E.selected=E.boxes.length-1;
    const valid=new Set(E.boxes.map(box=>box.id));E.selection=new Set([...E.selection].filter(id=>valid.has(id)));
    el('ann-boxes').innerHTML=boxRows().map(({box:b,index:i,number,invalid})=>`<div class="ann-box-row ${invalid?'out-of-bounds':''}" data-row-id="${b.id}" style="--box-color:${color(b)}"><button class="ann-order-handle" data-order-box="${b.id}" aria-label="拖动框 ${number} 调整序号" title="拖动排序，也可用上下箭头移动" ${E.busy||E.readOnly?'disabled':''}>⠿</button><button class="ann-box ${E.selection.has(b.id)?'selected':''} ${invalid?'out-of-bounds':''}" data-box="${i}" data-box-id="${b.id}"><strong>${number}. ${esc(b.label)}</strong><span class="ann-bound-flag" ${invalid?'':'hidden'}>越界，请检查</span><small>${Math.round(b.w)} × ${Math.round(b.h)} · ${(b.angle*180/Math.PI).toFixed(1)}°${b.score!=null?' · 预测 '+Number(b.score).toFixed(2):''}</small></button><button class="ann-row-delete" data-delete-box="${b.id}" aria-label="删除框 ${number}：${esc(b.label)}" title="删除此框" ${E.busy||E.readOnly?'disabled':''}><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18M9 6V4h6v2M5 6l1 14h12l1-14M10 10v6M14 10v6"/></svg></button></div>`).join('')||'<p>当前没有框</p>';
    el('ann-boxes').querySelectorAll('[data-box-id]').forEach(b=>b.onclick=event=>selectBox(b.dataset.boxId,event.shiftKey));
    el('ann-boxes').querySelectorAll('[data-delete-box]').forEach(b=>b.onclick=()=>removeBox(b.dataset.deleteBox));
    bindOrderHandles();el('ann-selection-count').textContent=E.selection.size?`已选 ${E.selection.size} 个框`:'';
    el('ann-delete').textContent=E.selection.size>1?`删除 ${E.selection.size} 个框`:'删除框';
    for(const id of ['ann-select','ann-marquee','ann-draw','ann-class','ann-copy','ann-save'])el(id).disabled=E.busy||E.readOnly;
    el('ann-copy').disabled||=E.index===0;
    el('ann-copy-box').disabled=E.busy||!E.image||E.selected<0;
    el('ann-paste-box').disabled=E.busy||E.readOnly||!clipboard||clipboard.shapes.some(shape=>!E.classes.includes(shape.label));
    el('ann-delete').disabled=E.busy||E.readOnly||!E.selection.size;
    el('ann-angle').disabled=E.busy||E.readOnly||E.selection.size!==1;
    el('ann-undo').disabled=E.busy||E.readOnly||!E.undo.length;el('ann-redo').disabled=E.busy||E.readOnly||!E.redo.length;
    el('ann-prev').disabled=E.busy||E.index===0;el('ann-next').disabled=E.busy||E.index===E.frames.length-1;el('ann-close').disabled=E.busy;
    el('ann-reload').disabled=E.busy;el('ann-next').textContent=E.readOnly?'下一张 →':'保存并下一张 →';el('ann-close').textContent=E.readOnly?'关闭':'保存并关闭';
    if(E.selected>=0){const b=E.boxes[E.selected];el('ann-class').value=b.label;el('ann-angle').value=(b.angle*180/Math.PI).toFixed(1)}else{el('ann-class').value=E.label;el('ann-angle').value=''}
    checkBounds();
  }
  function reorder(ids,targetId,after=false) {
    if(!available()||ids.includes(targetId))return;
    const moving=E.boxes.filter(box=>ids.includes(box.id)),remaining=E.boxes.filter(box=>!ids.includes(box.id));
    const index=remaining.findIndex(box=>box.id===targetId);if(!moving.length||index<0)return;
    remaining.splice(index+(after?1:0),0,...moving);
    if(remaining.every((box,i)=>box.id===E.boxes[i].id))return;
    snapshot();const primary=E.boxes[E.selected]?.id;E.boxes=remaining;selection(moving.map(box=>box.id),primary);changed();
    message('已调整框的顺序，画布序号同步更新；请保存。');
  }
  function bindOrderHandles() {
    el('ann-boxes').querySelectorAll('[data-order-box]').forEach(button=>{
      button.onpointerdown=event=>{
        if(!available()||event.button!==0)return;event.preventDefault();event.stopPropagation();
        const id=button.dataset.orderBox;const ids=E.selection.has(id)?[...E.selection]:[id];
        selection(ids,id);E.orderGesture={ids,start:[event.clientX,event.clientY],target:null,after:false,moved:false};
        button.setPointerCapture(event.pointerId);
        el('ann-boxes').querySelectorAll('[data-row-id]').forEach(row=>{const selected=ids.includes(row.dataset.rowId);row.classList.toggle('ordering',selected);row.querySelector('.ann-box').classList.toggle('selected',selected)});draw();
      };
      button.onpointermove=event=>{
        if(!E?.orderGesture)return;const g=E.orderGesture;if(Math.hypot(event.clientX-g.start[0],event.clientY-g.start[1])>5)g.moved=true;
        const list=el('ann-boxes'),rect=list.getBoundingClientRect();if(event.clientY<rect.top+24)list.scrollTop-=12;if(event.clientY>rect.bottom-24)list.scrollTop+=12;
        const row=document.elementFromPoint(event.clientX,event.clientY)?.closest('[data-row-id]');
        g.target=row&&list.contains(row)&&!g.ids.includes(row.dataset.rowId)?row.dataset.rowId:null;
        g.after=!!row&&event.clientY>row.getBoundingClientRect().top+row.getBoundingClientRect().height/2;
        list.querySelectorAll('[data-row-id]').forEach(item=>{item.classList.toggle('drop-before',item.dataset.rowId===g.target&&!g.after);item.classList.toggle('drop-after',item.dataset.rowId===g.target&&g.after)});
      };
      button.onpointerup=event=>{
        if(!E?.orderGesture)return;const g=E.orderGesture;E.orderGesture=null;
        if(button.hasPointerCapture(event.pointerId))button.releasePointerCapture(event.pointerId);
        if(g.moved&&g.target)reorder(g.ids,g.target,g.after);renderList();draw();
      };
      button.onpointercancel=()=>{if(E){E.orderGesture=null;renderList();draw()}};
      button.onkeydown=event=>{
        if(!['ArrowUp','ArrowDown'].includes(event.key))return;event.preventDefault();event.stopPropagation();if(!available())return;
        const id=button.dataset.orderBox,ids=E.selection.has(id)?[...E.selection]:[id],selected=E.boxes.filter(box=>ids.includes(box.id));
        const first=E.boxes.indexOf(selected[0]),last=E.boxes.indexOf(selected.at(-1));
        const target=E.boxes[event.key==='ArrowUp'?first-1:last+1];if(!target)return;
        selection(ids,id);reorder(ids,target.id,event.key==='ArrowDown');el('ann-boxes').querySelector(`[data-order-box="${id}"]`)?.focus();
      };
    });
  }
  function mode(value) {if(!E||E.gesture||E.orderGesture)return;E.mode=value;el('ann-draw').classList.toggle('primary',value==='draw');el('ann-marquee').classList.toggle('primary',value==='marquee');el('ann-select').classList.toggle('primary',value==='select');el('ann-canvas').style.cursor=value==='draw'||value==='marquee'?'crosshair':'default'}
  function pointerDown(event) {
    if(!E?.image||E.busy||E.orderGesture)return;const p=imagePoint(event);el('ann-canvas').focus();
    if(E.space||event.button===1){event.preventDefault();E.gesture={type:'pan',start:canvasPoint(event),offset:[...E.offset],before:clone(E.boxes)};el('ann-canvas').setPointerCapture(event.pointerId);return}
    if(!available()||event.button!==0)return;
    const before=clone(E.boxes);let gesture;
    if(E.mode==='marquee')gesture={type:'marquee',start:p,current:p,before,add:event.shiftKey,selectedBefore:[...E.selection]};
    else if(E.mode==='draw'){const start=p.map((v,i)=>Math.max(0,Math.min(i?E.frame.height:E.frame.width,v)));gesture={type:'draw',start,current:start,before}}
    else {
      const selected=E.boxes[E.selected];
      if(selected&&E.selection.size===1&&!event.shiftKey){const pts=corners(selected),corner=pts.findIndex(q=>Math.hypot(q[0]-p[0],q[1]-p[1])*E.scale<10);const hp=handle(selected);
        if(Math.hypot(hp[0]-p[0],hp[1]-p[1])*E.scale<12)gesture={type:'rotate',box:clone(selected),before,start:p};
        else if(corner>=0)gesture={type:'resize',box:clone(selected),before,opposite:pts[(corner+2)%4],start:p};}
      if(!gesture){let hit=-1;for(let i=E.boxes.length-1;i>=0;i--){const b=E.boxes[i],q=local(b,p);if(Math.abs(q[0])<=b.w/2&&Math.abs(q[1])<=b.h/2){hit=i;break}}
        if(hit>=0){const id=E.boxes[hit].id;if(event.shiftKey){selectBox(id,true);return}if(!E.selection.has(id))selection([id],id);else E.selected=hit;gesture={type:'move',box:clone(E.boxes[hit]),boxes:clone(selectedBoxes()),start:p,before}}
        else gesture={type:'marquee',start:p,current:p,before,add:event.shiftKey,selectedBefore:[...E.selection]};}
    }
    E.gesture=gesture;renderList();draw();if(gesture)el('ann-canvas').setPointerCapture(event.pointerId);
  }
  function pointerMove(event) {
    if(!E?.gesture)return;const g=E.gesture,p=imagePoint(event);
    if(g.type==='pan'){const pos=canvasPoint(event);E.offset=g.offset.map((v,i)=>v+pos[i]-g.start[i]);draw();return}
    if(g.type==='marquee')g.current=p;
    else if(g.type==='draw')g.current=p.map((v,i)=>Math.max(0,Math.min(i?E.frame.height:E.frame.width,v)));
    else {const b=E.boxes[E.selected];if(!b)return;
      if(g.type==='move'){const pts=g.boxes.flatMap(corners),dx=Math.max(-Math.min(...pts.map(q=>q[0])),Math.min(E.frame.width-Math.max(...pts.map(q=>q[0])),p[0]-g.start[0])),dy=Math.max(-Math.min(...pts.map(q=>q[1])),Math.min(E.frame.height-Math.max(...pts.map(q=>q[1])),p[1]-g.start[1]));for(const original of g.boxes){const moving=E.boxes.find(box=>box.id===original.id);moving.cx=original.cx+dx;moving.cy=original.cy+dy;if(original.original)moving.original=original.original.map(q=>[q[0]+dx,q[1]+dy])}}
      if(g.type==='rotate'){b.angle=Math.atan2(p[1]-b.cy,p[0]-b.cx)+Math.PI/2;if(event.shiftKey)b.angle=Math.round(b.angle/(Math.PI/12))*Math.PI/12;delete b.original}
      if(g.type==='resize'){const dx=p[0]-g.opposite[0],dy=p[1]-g.opposite[1],c=Math.cos(g.box.angle),s=Math.sin(g.box.angle);b.w=Math.max(1,Math.abs(dx*c+dy*s));b.h=Math.max(1,Math.abs(-dx*s+dy*c));b.cx=(p[0]+g.opposite[0])/2;b.cy=(p[1]+g.opposite[1])/2;delete b.original}
    }draw();
  }
  function pointerUp(event) {
    if(!E?.gesture)return;const g=E.gesture;E.gesture=null;
    if(el('ann-canvas').hasPointerCapture(event.pointerId))el('ann-canvas').releasePointerCapture(event.pointerId);
    if(g.type==='pan')return;
    if(g.type==='marquee'){const left=Math.min(g.start[0],g.current[0]),right=Math.max(g.start[0],g.current[0]),top=Math.min(g.start[1],g.current[1]),bottom=Math.max(g.start[1],g.current[1]);const ids=E.boxes.filter(box=>corners(box).every(p=>p[0]>=left&&p[0]<=right&&p[1]>=top&&p[1]<=bottom)).map(box=>box.id);selection([...new Set([...(g.add?g.selectedBefore:[]),...ids])]);mode('select');renderList();draw();message(`已选 ${E.selection.size} 个框，可一起移动或拖动右侧 ⠿ 排序。`);return}
    if(g.type==='draw'){const [a,b]=[g.start,g.current],w=Math.abs(b[0]-a[0]),h=Math.abs(b[1]-a[1]);if(w<1||h<1){draw();return}E.boxes.push({id:boxId(),label:E.label,cx:(a[0]+b[0])/2,cy:(a[1]+b[1])/2,w,h,angle:0});selection([E.boxes.at(-1).id])}
    if(JSON.stringify(g.before)!==JSON.stringify(E.boxes)){E.undo.push(g.before);if(E.undo.length>80)E.undo.shift();E.redo=[];changed()}else{renderList();draw()}
  }
  function removeBox(id) {
    if(!available())return;const index=E.boxes.findIndex(box=>box.id===id);if(index<0)return;
    const selectedId=E.boxes[E.selected]?.id;snapshot();E.boxes.splice(index,1);
    const ids=[...E.selection].filter(value=>value!==id);if(selectedId===id&&!ids.length&&E.boxes.length)ids.push(E.boxes[Math.min(index,E.boxes.length-1)].id);selection(ids,ids.includes(selectedId)?selectedId:ids.at(-1));changed();
  }
  function remove() {if(!available()||!E.selection.size)return;if(E.selection.size===1)return removeBox([...E.selection][0]);snapshot();E.boxes=E.boxes.filter(box=>!E.selection.has(box.id));selection([]);changed()}
  function history(direction) {if(!available()||!E[direction].length)return;const other=direction==='undo'?'redo':'undo';E[other].push(clone(E.boxes));E.boxes=E[direction].pop();selection([]);changed()}
  function copyBox() {
    if(!E?.image||E.busy||E.gesture||E.selected<0)return;const box=E.boxes[E.selected];
    const boxes=selectedBoxes();clipboard={label:box.label,shapes:boxes.map(b=>({label:b.label,points:clone(corners(b))})),width:E.frame.width,height:E.frame.height,source:key(),pasteCount:0};
    renderList();message(`已复制 ${boxes.length} 个框，Ctrl+V 粘贴。`);
  }
  function pasteBox() {
    if(!available()||!E.image||!clipboard)return;
    if(clipboard.shapes.some(shape=>!E.classes.includes(shape.label)))throw new Error('复制框的类别不属于当前实验，请复制当前实验的框。');
    const same=clipboard.source===key(),offset=same?12*(clipboard.pasteCount+1):0;
    const shapes=clipboard.shapes.map(shape=>({...shape,points:shape.points.map(p=>[p[0]*E.frame.width/clipboard.width,p[1]*E.frame.height/clipboard.height])})),pts=shapes.flatMap(shape=>shape.points);
    if(offset){const dx=Math.max(0,Math.min(offset,E.frame.width-Math.max(...pts.map(p=>p[0])))),dy=Math.max(0,Math.min(offset,E.frame.height-Math.max(...pts.map(p=>p[1]))));shapes.forEach(shape=>shape.points=shape.points.map(p=>[p[0]+dx,p[1]+dy]))}
    const pasted=shapes.map(fromShape);snapshot();E.boxes.push(...pasted);selection(pasted.map(box=>box.id));clipboard.pasteCount++;mode('select');changed();
    message(`已粘贴 ${pasted.length} 个框，请调整位置后保存。`);
  }
  async function copyPrevious() {
    if(!available()||E.index===0)return;setBusy(true);
    try{const prev=E.frames[E.index-1],data=await fetchAnnotation(prev);if(!data.saved)throw new Error('上一张还没有保存标注。');const shapes=data.document.shapes.map(shape=>({...shape,points:shape.points.map(p=>[p[0]*E.frame.width/prev.width,p[1]*E.frame.height/prev.height])}));
      if(!shapes.length)throw new Error('上一张没有框可复制。');snapshot();const pasted=shapes.map(fromShape);E.boxes.push(...pasted);selection(pasted.map(box=>box.id));changed();message(`已追加上一张的 ${shapes.length} 个框，请根据当前图片调整并保存。`);
    }finally{setBusy(false)}
  }
  async function save() {
    if(!E||E.readOnly||E.busy)return false;if(E.gesture||E.orderGesture)throw new Error('请先完成当前拖动操作。');
    const shapes=E.boxes.map(box=>({label:box.label,points:corners(box),shape_type:'rotation',flags:{}}));
    const invalid=boxRows().filter(row=>row.invalid);if(invalid.length){focusBox(invalid[0].box.id);throw new Error(`框 ${invalid.map(row=>row.number).join('、')} 超出图片边界，已高亮，请调整后保存。`)}
    setBusy(true);
    try{const result=await api('/api/annotation',{project_id:E.projectId,frame:E.frame.id,subset:E.round?undefined:E.subset,round:E.round||undefined,revision:E.revision,shapes});E.revision=result.revision;E.dirty=false;const ids=E.boxes.map(b=>b.id);E.boxes=result.document.shapes.map((shape,i)=>({...fromShape(shape),id:ids[i]}));draft();message(E.round?'已保存并确认此张审核。':'已保存标注。');return true}
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
    if((event.ctrlKey||event.metaKey)&&key==='c'){event.preventDefault();copyBox()}
    else if((event.ctrlKey||event.metaKey)&&key==='v'){event.preventDefault();action(pasteBox)}
    else if((event.ctrlKey||event.metaKey)&&['z','y'].includes(key)){event.preventDefault();history(key==='y'||event.shiftKey?'redo':'undo')}
    else if(key==='d')mode('draw');else if(key==='v')mode('select');else if(key==='m')mode('marquee');else if(key==='c')action(copyPrevious);
    else if(event.key==='Delete'||event.key==='Backspace'){event.preventDefault();remove()}
    else if(event.key==='ArrowRight'){event.preventDefault();action(()=>navigate(1))}else if(event.key==='ArrowLeft'){event.preventDefault();action(()=>navigate(-1))}
  });
  document.addEventListener('keyup',event=>{if(E&&event.code==='Space')E.space=false});
  window.addEventListener('blur',()=>{if(E)E.space=false});
  return {open, corners};
})();
