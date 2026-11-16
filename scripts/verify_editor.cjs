// Browser regressions. Install Playwright separately; use the project's basic Python environment.
const {chromium}=require('playwright');
const {spawn,execFileSync}=require('node:child_process');
const fs=require('node:fs');const path=require('node:path');const assert=require('node:assert/strict');

(async()=>{
  const workspace=path.resolve('_smoke/editor_regression_'+Date.now());
  fs.mkdirSync(workspace,{recursive:true});
  const python=process.env.ALAB_TEST_PYTHON||path.resolve('.venv/Scripts/python.exe');
  const env={...process.env,PYTHONPATH:path.resolve('src'),PYTHONIOENCODING:'utf-8'};
  execFileSync(python,['-c',String.raw`
import sys
from pathlib import Path
import cv2,numpy as np
from autolabel_lab.project import Project,fingerprint,sha256,write_json
from autolabel_lab.labels import make_document
p=Project.create(Path(sys.argv[1])/'experiments/editor',['target','other'])
frames=[]
for i,role in enumerate(['seed','seed','val','test','pool']):
    w,h=(960,640) if i==1 else (480,320)
    pixels=np.full((h,w,3),235,np.uint8)
    image=p.root/'frames'/f'f{i}.png'
    cv2.putText(pixels,f'Frame {i}',(20,h-30),cv2.FONT_HERSHEY_SIMPLEX,1,(40,90,70),2)
    image.write_bytes(cv2.imencode('.png',pixels)[1].tobytes())
    frames.append({'id':f'f{i}','file':image.name,'role':role,'width':w,'height':h,'image_sha256':sha256(image),
                   'frame_index':i,'timestamp':i,'source_kind':'image'})
write_json(p.root/'manifest.json',{'frames':frames,'split':{'frames_hash':fingerprint(frames)}})
f=frames[-1]
doc=make_document(f,[{'label':'target','points':[[-20,40],[40,40],[40,80],[-20,80]],'score':.8},
                     {'label':'other','points':[[520,100],[560,100],[560,160],[520,160]],'score':.9}],prediction=True)
raw=p.root/'rounds/r1/raw/f4.json'
write_json(raw,doc);write_json(p.root/'rounds/r1/editable/f4.json',doc)
write_json(p.root/'rounds/r1/round.json',{'name':'r1','model':'M0','status':'predicted','frame_ids':['f4'],
                                    'raw_hashes':{'f4':sha256(raw)},'params':{'conf':.25,'imgsz':640}})
`,workspace],{env,windowsHide:true});
  // Port 0 prevents conflicts with running workbenches and writes the chosen port to disk.
  const server=spawn(python,['-c',String.raw`
import sys
from pathlib import Path
from autolabel_lab.web import make_server
s=make_server(sys.argv[1],port=0)
(Path(sys.argv[1])/'port.txt').write_text(str(s.server_port))
s.serve_forever()
`,workspace],{env,windowsHide:true,stdio:'ignore'});
  let browser;const checks=[],errors=[];
  try{
    for(let i=0;i<100&&!fs.existsSync(path.join(workspace,'port.txt'));i++)await new Promise(r=>setTimeout(r,100));
    const base='http://127.0.0.1:'+fs.readFileSync(path.join(workspace,'port.txt'),'utf8');
    browser=await chromium.launch({channel:'msedge',headless:true});
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(base);await page.waitForFunction(()=>typeof S!=='undefined'&&S.details);
    const id=await page.evaluate(()=>S.id);
    const read=async(frame,subset='seed',round='')=>(await(await fetch(base+`/api/annotation?id=${id}&frame=${frame}&${round?'round='+round:'subset='+subset}`)).json());
    await page.locator('#nav [data-page=label]').click();await page.locator('[data-annotate-subset=seed]').first().click();
    await page.waitForFunction(()=>document.querySelector('#ann-status').textContent.includes('尚未标注'));
    // Record canvas label text without relying on screenshot OCR.
    await page.evaluate(()=>{const canvas=document.querySelector('#ann-canvas'),ctx=canvas.getContext('2d'),fill=ctx.fillText.bind(ctx),transform=ctx.setTransform.bind(ctx);window.drawnLabels=[];ctx.setTransform=(...a)=>{window.drawnLabels=[];transform(...a)};ctx.fillText=(text,...a)=>{window.drawnLabels.push(text);fill(text,...a)}});
    const point=async(x,y)=>page.evaluate(([x,y])=>{const r=document.querySelector('#ann-canvas').getBoundingClientRect(),s=Math.min(r.width/480,r.height/320)*.95;return {x:r.x+(r.width-480*s)/2+x*s,y:r.y+(r.height-320*s)/2+y*s}},[x,y]);
    const drag=async(a,b)=>{await page.mouse.move(a.x,a.y);await page.mouse.down();await page.mouse.move(b.x,b.y,{steps:5});await page.mouse.up()};
    await page.locator('#ann-draw').click();
    for(const [i,x] of [60,180,300].entries()){
      await drag(await point(x,80),await point(x+60,130));
    }
    const row=page.locator('.ann-box');
    await page.locator('.ann-display summary').click();
    assert.equal(await page.locator('#ann-background').inputValue(),'65');
    await page.selectOption('#ann-label-mode','selected');
    assert.deepEqual(await page.evaluate(()=>window.drawnLabels),['3. target']);
    await page.selectOption('#ann-label-mode','hidden');assert.deepEqual(await page.evaluate(()=>window.drawnLabels),[]);
    await page.selectOption('#ann-label-mode','all');
    assert.equal((await page.evaluate(()=>window.drawnLabels)).length,3);
    await page.locator('#ann-background').fill('80');await page.locator('#ann-background').dispatchEvent('input');
    await page.locator('#ann-font-size').fill('16');await page.locator('#ann-font-size').dispatchEvent('input');
    await page.locator('#ann-line-width').fill('2.6');await page.locator('#ann-line-width').dispatchEvent('input');
    assert.equal(await page.locator('#ann-background-value').innerText(),'80%');
    const prefs=await page.evaluate(()=>JSON.parse(localStorage.getItem('alab-annotation-display')));
    assert(Math.abs(prefs.backgroundOpacity-.2)<1e-9);assert.equal(prefs.fontSize,16);assert.equal(prefs.lineWidth,2.6);
    const drawing=await page.evaluate(()=>{const canvas=document.createElement('canvas');canvas.width=200;canvas.height=80;const ctx=canvas.getContext('2d'),calls=[];for(const method of ['fillRect','strokeText','fillText']){const original=ctx[method].bind(ctx);ctx[method]=(...args)=>{calls.push({method,alpha:ctx.globalAlpha,font:ctx.font});original(...args)}}drawAnnotationLabel(ctx,'Label',[10,40],'#28685e',200,80,{backgroundOpacity:.2,fontSize:16});return calls});
    assert(Math.abs(drawing.find(c=>c.method==='fillRect').alpha-.2)<1e-6);
    assert.equal(drawing.find(c=>c.method==='fillText').alpha,1);assert.equal(drawing.find(c=>c.method==='strokeText').alpha,1);
    await page.locator('#ann-display-reset').click();assert.equal(await page.locator('#ann-font-size').inputValue(),'12');
    checks.push('label transparency only affects background, outlined opaque text and selectable visibility');
    const initialIds=await row.evaluateAll(buttons=>buttons.map(button=>button.dataset.boxId));
    await page.locator('#ann-marquee').click();await drag(await point(50,70),await point(250,145));
    assert.equal(await page.locator('.ann-box.selected').count(),2);
    await drag(await point(90,105),await point(100,115));
    await page.locator('#ann-save').click();await page.waitForFunction(()=>document.querySelector('#ann-status').textContent.includes('已保存'));
    const moved=(await read('f0')).document.shapes;
    assert(Math.abs(moved[0].points[0][0]-70)<.01);assert(Math.abs(moved[1].points[0][0]-190)<.01);assert(Math.abs(moved[2].points[0][0]-300)<.01);
    await page.locator('#ann-copy-box').click();await page.locator('#ann-paste-box').click();assert.equal(await row.count(),5);
    assert.equal(await page.locator('.ann-box.selected').count(),2);await page.locator('#ann-delete').click();assert.equal(await row.count(),3);
    await page.locator('#ann-undo').click();assert.equal(await row.count(),5);await page.locator('#ann-undo').click();assert.equal(await row.count(),3);
    await page.locator('#ann-undo').click();await page.locator('#ann-save').click();await page.waitForFunction(()=>document.querySelector('#ann-status').textContent.includes('已保存'));
    const handle=page.locator('[data-order-box]');
    let from=await handle.nth(2).boundingBox(),target=await page.locator('[data-row-id]').first().boundingBox();
    await drag({x:from.x+from.width/2,y:from.y+from.height/2},{x:target.x+target.width/2,y:target.y+2});
    assert.deepEqual(await row.evaluateAll(buttons=>buttons.map(button=>button.dataset.boxId)),[initialIds[2],initialIds[0],initialIds[1]]);
    assert.deepEqual(await page.evaluate(()=>window.drawnLabels),['1. target','2. target','3. target']);
    await page.locator('#ann-save').click();await page.waitForFunction(()=>document.querySelector('#ann-status').textContent.includes('已保存'));
    const reordered=(await read('f0')).document.shapes;assert(reordered[0].points.every(p=>p[0]>=300));
    await page.locator('#ann-undo').click();assert.deepEqual(await row.evaluateAll(buttons=>buttons.map(button=>button.dataset.boxId)),initialIds);
    await row.first().click();await row.nth(2).click({modifiers:['Shift']});assert.equal(await page.locator('.ann-box.selected').count(),2);
    from=await handle.first().boundingBox();target=await page.locator('[data-row-id]').nth(1).boundingBox();
    await drag({x:from.x+from.width/2,y:from.y+from.height/2},{x:target.x+target.width/2,y:target.y+target.height-2});
    assert.deepEqual(await row.evaluateAll(buttons=>buttons.map(button=>button.dataset.boxId)),[initialIds[1],initialIds[0],initialIds[2]]);
    assert.equal(await page.locator('.ann-box.selected').count(),2);await page.locator('#ann-undo').click();
    await handle.nth(2).focus();await page.keyboard.press('ArrowUp');assert.deepEqual(await row.evaluateAll(buttons=>buttons.map(button=>button.dataset.boxId)),[initialIds[0],initialIds[2],initialIds[1]]);
    await page.locator('#ann-undo').click();await page.locator('#ann-save').click();await page.waitForFunction(()=>document.querySelector('#ann-status').textContent.includes('已保存'));
    assert.deepEqual(await row.evaluateAll(buttons=>buttons.map(button=>button.dataset.boxId)),initialIds);
    checks.push('marquee selection and group movement, multi-copy/delete undo', 'single/group drag reorder, keyboard ordering and persisted sequence');
    await row.nth(2).click();const selectedLast=await row.nth(2).getAttribute('data-box-id');
    await page.locator('[data-delete-box]').nth(1).click();
    assert.deepEqual(await row.locator('strong').allTextContents(),['1. target','2. target']);
    assert.deepEqual(await page.evaluate(()=>window.drawnLabels),['1. target','2. target']);
    assert.equal(await page.locator('.ann-box.selected').getAttribute('data-box'),'1');
    assert.equal(await page.locator('.ann-box.selected').getAttribute('data-box-id'),selectedLast);
    await page.selectOption('#ann-class','other');await page.locator('#ann-save').click();
    await page.waitForFunction(()=>document.querySelector('#ann-status').textContent.includes('已保存'));
    let saved=(await read('f0')).document.shapes;
    assert.equal(saved.length,2);assert.equal(saved[1].label,'other');assert(saved[1].points.every(p=>p[0]>=300));
    await page.locator('[data-delete-box]').first().click();
    assert.deepEqual(await row.locator('strong').allTextContents(),['1. other']);
    assert.deepEqual(await page.evaluate(()=>window.drawnLabels),['1. other']);
    assert.equal(await page.locator('.ann-box.selected').getAttribute('data-box'),'0');
    assert.equal(await page.locator('.ann-box.selected').getAttribute('data-box-id'),selectedLast);
    assert.equal(await page.locator('[data-delete-box]').first().getAttribute('aria-label'),'删除框 1：other');
    await page.locator('#ann-undo').click();assert.equal(await row.count(),2);
    await page.locator('#ann-redo').click();assert.equal(await row.count(),1);
    await row.first().click();await page.locator('#ann-angle').fill('20');await page.locator('#ann-angle').dispatchEvent('change');
    await page.locator('#ann-canvas').focus();await page.keyboard.press('Control+c');await page.keyboard.press('Control+v');
    assert.equal(await row.count(),2);assert.deepEqual(await row.locator('strong').allTextContents(),['1. other','2. other']);
    await page.locator('#ann-save').click();await page.waitForFunction(()=>document.querySelector('#ann-status').textContent.includes('已保存'));
    saved=(await read('f0')).document.shapes;assert.equal(saved.length,2);
    assert(Math.abs(saved[1].points[0][0]-saved[0].points[0][0]-12)<.01);
    assert(Math.abs(saved[1].points[0][1]-saved[0].points[0][1]-12)<.01);
    await page.locator('#ann-copy-box').click();await page.locator('#ann-next').click();
    await page.waitForFunction(()=>document.querySelector('#ann-position').textContent==='2 / 2');
    assert.equal(await page.locator('#ann-background').inputValue(),'65');
    await page.locator('#ann-paste-box').click();assert.equal(await row.count(),1);
    await page.locator('#ann-save').click();await page.waitForFunction(()=>document.querySelector('#ann-status').textContent.includes('已保存'));
    const pasted=(await read('f1')).document.shapes[0];assert.equal(pasted.label,'other');
    for(let i=0;i<4;i++)for(let axis=0;axis<2;axis++)assert(Math.abs(pasted.points[i][axis]-saved[1].points[i][axis]*2)<.01);
    // Ctrl+C/V in a text field must not operate on the selected annotation.
    await row.first().click();await page.locator('#ann-angle').focus();await page.keyboard.press('Control+v');assert.equal(await row.count(),1);
    await page.locator('#ann-canvas').focus();await page.keyboard.press('Control+v');assert.equal(await row.count(),2);
    await page.locator('#ann-undo').click();assert.equal(await row.count(),1);
    await row.first().click();await page.locator('[data-delete-box]').first().click();assert.equal(await row.count(),0);assert(await page.locator('#ann-copy-box').isDisabled());
    await page.locator('#ann-undo').click();await page.locator('#ann-close').click();
    const boot=await(await fetch(base+'/api/bootstrap')).json();
    const importResponse=await fetch(base+'/api/job',{method:'POST',headers:{'Content-Type':'application/json','X-Lab-Token':boot.token},body:JSON.stringify({project_id:id,action:'import-labels',payload:{subset:'seed'}})});
    assert.equal(importResponse.status,200);const job=await importResponse.json();
    for(let i=0;i<100;i++){const state=await(await fetch(base+'/api/overview')).json();const current=state.jobs.find(j=>j.id===job.id);if(!['queued','running','recovering'].includes(current.status)){assert.equal(current.status,'completed');break;}await page.waitForTimeout(100);if(i===99)throw new Error('Archive job timed out')}
    await page.evaluate(()=>refresh(true));await page.locator('[data-annotate-subset=seed]').first().click();
    await page.waitForFunction(()=>document.querySelector('#ann-title').textContent.includes('只读'));
    assert.equal(await page.locator('[data-delete-box]').count(),2);
    for(const button of await page.locator('[data-delete-box]').all())assert(await button.isDisabled());
    for(const button of await page.locator('[data-order-box]').all())assert(await button.isDisabled());
    await page.locator('#ann-close').click();
    checks.push('sidebar deletes first/middle/last box, keeps other selection and correct numbering', 'row deletion undo, saved result and archived read-only protection', 'copy/paste via shortcuts and buttons, rotation, cross-image scaling and undo');
    await page.locator('#nav [data-page=auto]').click();await page.locator('[data-annotate-round=r1]').click();
    await page.waitForFunction(()=>document.querySelectorAll('.ann-box').length===2);
    assert.equal(await page.locator('.ann-box.out-of-bounds').count(),2);
    assert.equal(await page.locator('.ann-check-box').count(),2);
    assert(await page.locator('#ann-bounds').isVisible());
    await page.locator('.ann-display summary').click();await page.locator('#ann-background').fill('80');await page.locator('#ann-background').dispatchEvent('input');
    await page.selectOption('#ann-label-mode','hidden');
    const rawLabels=await page.evaluate(()=>{const canvas=document.querySelector('#ann-raw'),ctx=canvas.getContext('2d'),original=ctx.fillText.bind(ctx);let count=0;ctx.fillText=(...args)=>{count++;original(...args)};document.querySelector('#ann-background').dispatchEvent(new Event('input'));ctx.fillText=original;return count});
    assert.equal(rawLabels,0);assert.equal(await page.locator('.ann-box.out-of-bounds').count(),2);
    await page.selectOption('#ann-label-mode','all');
    const raw=(await read('f4','','r1')).raw;
    await page.locator('[data-check-box]').nth(1).click();
    assert.equal(await page.locator('.ann-box.selected').getAttribute('data-box'),'1');
    await page.locator('#ann-save').click();await page.waitForFunction(()=>document.querySelector('#ann-status').textContent.includes('框 1、2'));
    assert.equal(await page.locator('.ann-box.selected').getAttribute('data-box'),'0');
    await page.screenshot({path:path.join(workspace,'bounds-warning.png')});
    // Repair a partly out-of-bounds rectangle by dragging it into the image.
    await page.locator('#ann-select').click();await page.locator('#ann-fit').click();
    await drag(await point(10,60),await point(65,60));
    assert.equal(await page.locator('.ann-box.out-of-bounds').count(),1);
    assert.equal(await page.locator('.ann-check-box').count(),1);
    await page.locator('[data-check-box]').click();await page.locator('[data-delete-box]').nth(1).click();
    assert(await page.locator('#ann-bounds').isHidden());assert.equal(await page.locator('.ann-box.out-of-bounds').count(),0);
    await page.locator('#ann-save').click();await page.waitForFunction(()=>document.querySelector('#ann-status').textContent.includes('已保存'));
    const result=await read('f4','','r1');assert.deepEqual(result.raw,raw);assert.equal(result.document.shapes.length,1);
    await page.locator('#ann-copy-box').click();await page.locator('#ann-paste-box').click();
    for(const width of [1440,768,390]){await page.setViewportSize({width,height:1000});await page.waitForTimeout(100);assert(!(await page.evaluate(()=>document.querySelector('#annotation-dialog').scrollWidth>innerWidth+2)))}
    await page.locator('#ann-close').click();await page.reload();await page.waitForFunction(()=>typeof S!=='undefined'&&S.details);await page.locator('#nav [data-page=auto]').click();await page.locator('[data-annotate-round=r1]').click();await page.waitForFunction(()=>document.querySelector('#ann-position').textContent==='1 / 1');assert.equal(await page.locator('#ann-background').inputValue(),'80');
    checks.push('out-of-bounds load highlighting, off-image focus, save blocking, repair and raw preservation','responsive toolbar and display settings persist after reload');
    assert.deepEqual(errors,[]);fs.writeFileSync(path.join(workspace,'verification.json'),JSON.stringify({checks,errors},null,2));console.log(JSON.stringify({workspace,checks,errors},null,2));
  }finally{if(browser)await browser.close();server.kill()}
})().catch(error=>{console.error(error);process.exitCode=1});
