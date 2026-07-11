"""Human review UI builder.

Selects annotated episodes (half of them containing review-flagged subtasks,
half clean — so the flag calibration question gets answered), transcodes each
episode video to a small browser-playable H.264 MP4, and emits ONE
self-contained review page: segment timeline synced to the video, per-segment
play (with ±0.5 s pre/post-roll for boundary judgment), verdict buttons per
segment (boundary / label / sentence) plus an episode-level coherence verdict.
Verdicts persist in localStorage; the Export button downloads a JSON that
eval/report_verdicts.py scores.

    python make_review.py --category pick_place --n 20 [--seed 11]
    -> outputs/review/index.html   (open in any browser)
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config, load_task_categories  # noqa: E402


def select_episodes(cfg, category, n, seed):
    ann_dir = Path(cfg["paths"]["output_dir"]) / "annotations"
    anns = []
    for f in sorted(ann_dir.glob("*.json")):
        a = json.loads(f.read_text())
        if category:
            categories, task_map = load_task_categories()
            if task_map.get(a["episode_id"].split("/")[0]) != category:
                continue
        anns.append(a)
    if not anns:
        raise SystemExit("no annotated episodes match the selection")

    rng = random.Random(seed)
    flagged = [a for a in anns if a["review_queue"]]
    clean = [a for a in anns if not a["review_queue"]]
    rng.shuffle(flagged)
    rng.shuffle(clean)

    picked, seen_tasks = [], set()
    # alternate flagged/clean, preferring unseen tasks for spread
    pools = [flagged, clean]
    while len(picked) < n and any(pools):
        for pool in pools:
            if len(picked) >= n:
                break
            pool.sort(key=lambda a: a["episode_id"].split("/")[0] in seen_tasks)
            if pool:
                a = pool.pop(0)
                picked.append(a)
                seen_tasks.add(a["episode_id"].split("/")[0])
    return picked


def transcode(src, dst):
    if dst.exists():
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vf", "scale=640:-2", "-c:v", "libx264", "-crf", "26",
         "-preset", "veryfast", "-an", "-movflags", "+faststart", str(dst)],
        check=True)


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>EgoDex annotation review</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh}
 #side{width:270px;overflow-y:auto;border-right:1px solid #ccc;padding:8px;flex-shrink:0}
 #side .ep{padding:5px 7px;border-radius:5px;cursor:pointer;font-size:13px;margin:2px 0}
 #side .ep:hover{background:#eef}
 #side .ep.active{background:#dde6ff;font-weight:600}
 #side .ep .st{float:right;font-size:11px}
 #main{flex:1;overflow-y:auto;padding:14px 22px}
 video{width:640px;max-width:100%;background:#000;border-radius:6px}
 #timeline{position:relative;width:640px;max-width:100%;height:34px;margin:6px 0 2px;
   background:#f2f2f2;border-radius:5px;overflow:hidden;cursor:pointer}
 #timeline .blk{position:absolute;top:0;height:100%;border-right:2px solid #fff;
   font-size:10px;color:#fff;overflow:hidden;padding:2px 3px;box-sizing:border-box}
 #timeline .cursor{position:absolute;top:0;width:2px;height:100%;background:red}
 #caption{width:640px;max-width:100%;min-height:38px;font-size:15px;padding:6px 2px}
 table{border-collapse:collapse;margin-top:8px;width:100%}
 td,th{border-bottom:1px solid #e3e3e3;padding:7px 8px;font-size:13px;text-align:left;vertical-align:top}
 tr.active{background:#f4f7ff}
 .vbtn{border:1px solid #bbb;background:#fff;border-radius:4px;padding:2px 9px;cursor:pointer;margin:0 2px;font-size:13px}
 .vbtn.good{background:#2e9e44;color:#fff;border-color:#2e9e44}
 .vbtn.bad{background:#d33;color:#fff;border-color:#d33}
 .play{cursor:pointer;color:#25c;font-weight:600;white-space:nowrap}
 .flag{color:#c60;font-size:11px}
 #epverdict{margin:10px 0;padding:10px;background:#f8f8f8;border-radius:6px;width:640px;max-width:100%;box-sizing:border-box}
 input[type=text]{width:220px;font-size:12px}
 #topbar{position:sticky;top:0;background:#fff;padding:6px 0;z-index:5;border-bottom:1px solid #eee}
 button.big{padding:6px 14px;font-size:14px;border-radius:6px;border:1px solid #888;cursor:pointer}
 .done{color:#2e9e44}.part{color:#c90}
 .hint{color:#888;font-size:12px}
</style></head><body>
<div id="side"><h3 style="margin:4px">Episodes</h3><div id="eplist"></div></div>
<div id="main">
 <div id="topbar">
  <button class="big" id="prev">&larr; prev</button>
  <button class="big" id="next">next &rarr;</button>
  <button class="big" id="export">Export verdicts JSON</button>
  <span class="hint" id="progress"></span>
 </div>
 <h2 id="title"></h2><div id="task" class="hint"></div>
 <video id="vid" controls preload="metadata"></video>
 <div id="timeline"></div><div id="caption"></div>
 <div id="epverdict"></div>
 <table id="segtable"></table>
 <p class="hint">Per segment: <b>&#9654; play</b> plays the segment with 0.5 s
 pre/post-roll (judge the cut points). <b>Boundary</b> = start/end cuts correct,
 nothing missed inside. <b>Label</b> = action+hand+object correct.
 <b>Sentence</b> = description factually right. Verdicts save automatically
 (localStorage); export when done.</p>
</div>
<script>
const DATA = %%DATA%%;
const LS = 'egodex_review_v1';
let store = JSON.parse(localStorage.getItem(LS) || '{}');
let cur = 0, loopEnd = null;

const vid = document.getElementById('vid');
const $ = id => document.getElementById(id);

function v(ep){ if(!store[ep.id]) store[ep.id] = {episode_ok:null, note:'', segments:{}}; return store[ep.id]; }
function sv(ep, sid){ const e = v(ep); if(!e.segments[sid]) e.segments[sid] = {boundary:null,label:null,sentence:null,note:''}; return e.segments[sid]; }
function save(){ localStorage.setItem(LS, JSON.stringify(store)); renderSide(); renderProgress(); }

function epState(ep){
  const e = store[ep.id]; if(!e) return '';
  const total = ep.subtasks.length*3 + 1;
  let done = e.episode_ok===null?0:1;
  for(const s of ep.subtasks){ const g=(e.segments||{})[s.id]; if(g){ for(const k of ['boundary','label','sentence']) if(g[k]!==null) done++; } }
  return done===0?'':(done===total?'&#10003;':'&#9683;');
}
function renderSide(){
  $('eplist').innerHTML = DATA.map((ep,i)=>
    `<div class="ep ${i===cur?'active':''}" onclick="go(${i})">${ep.id}
     <span class="st ${epState(ep)==='&#10003;'?'done':'part'}">${epState(ep)}</span></div>`).join('');
}
function renderProgress(){
  const done = DATA.filter(ep=>epState(ep)==='&#10003;').length;
  $('progress').textContent = ` reviewed ${done}/${DATA.length} episodes`;
}
function btn(cls, state, want, label, cb){
  return `<button class="vbtn ${state===want?(want?'good':'bad'):''}" onclick='${cb}'>${label}</button>`;
}
function renderMain(){
  const ep = DATA[cur];
  $('title').textContent = ep.id;
  $('task').textContent = 'Task: ' + ep.task;
  vid.src = 'media/' + ep.file; loopEnd = null;
  const dur = ep.duration;
  $('timeline').innerHTML = ep.subtasks.map((s,i)=>
    `<div class="blk" style="left:${100*s.start_time/dur}%;width:${100*(s.end_time-s.start_time)/dur}%;
      background:hsl(${(i*67)%360},55%,48%)" onclick="playSeg(${i});event.stopPropagation()">${s.action}</div>`
   ).join('') + '<div class="cursor" id="cursor"></div>';
  const e = v(ep);
  $('epverdict').innerHTML = `<b>Episode-level: is the subtask sequence coherent and complete?</b>
    ${btn('',e.episode_ok,true,'Coherent',`epv(true)`)} ${btn('',e.episode_ok,false,'Issues',`epv(false)`)}
    note: <input type="text" value="${(e.note||'').replace(/"/g,'&quot;')}" onchange="epnote(this.value)">`;
  $('segtable').innerHTML = '<tr><th></th><th>t</th><th>action</th><th>hand</th><th>object</th><th>sentence</th><th>boundary</th><th>label</th><th>sentence ok</th><th>note</th></tr>' +
    ep.subtasks.map(s=>{
      const g = sv(ep, s.id);
      return `<tr id="row${s.id}">
       <td class="play" onclick="playSeg(${s.id})">&#9654; #${s.id}</td>
       <td>${s.start_time.toFixed(1)}&ndash;${s.end_time.toFixed(1)}s</td>
       <td><b>${s.action}</b>${s.flags.length?`<div class="flag">${s.flags.join(',')}</div>`:''}</td>
       <td>${s.hand}</td><td>${s.object}</td><td style="max-width:290px">${s.subtask}</td>
       <td>${btn('',g.boundary,true,'&#10003;',`segv(${s.id},'boundary',true)`)}${btn('',g.boundary,false,'&#10007;',`segv(${s.id},'boundary',false)`)}</td>
       <td>${btn('',g.label,true,'&#10003;',`segv(${s.id},'label',true)`)}${btn('',g.label,false,'&#10007;',`segv(${s.id},'label',false)`)}</td>
       <td>${btn('',g.sentence,true,'&#10003;',`segv(${s.id},'sentence',true)`)}${btn('',g.sentence,false,'&#10007;',`segv(${s.id},'sentence',false)`)}</td>
       <td><input type="text" value="${(g.note||'').replace(/"/g,'&quot;')}" onchange="segnote(${s.id},this.value)"></td>
      </tr>`;}).join('');
  renderSide(); renderProgress();
}
window.go = i => { cur = i; renderMain(); };
window.epv = val => { const e=v(DATA[cur]); e.episode_ok = (e.episode_ok===val)?null:val; save(); renderMain(); };
window.epnote = val => { v(DATA[cur]).note = val; save(); };
window.segv = (sid, field, val) => { const g=sv(DATA[cur],sid); g[field]=(g[field]===val)?null:val; save(); renderMain(); };
window.segnote = (sid, val) => { sv(DATA[cur],sid).note = val; save(); };
window.playSeg = sid => {
  const s = DATA[cur].subtasks[sid];
  vid.currentTime = Math.max(0, s.start_time - 0.5);
  loopEnd = Math.min(DATA[cur].duration, s.end_time + 0.5);
  vid.play();
  document.querySelectorAll('#segtable tr').forEach(r=>r.classList.remove('active'));
  const row = $('row'+sid); if(row) row.classList.add('active');
};
vid.addEventListener('timeupdate', ()=>{
  const ep = DATA[cur];
  if(loopEnd !== null && vid.currentTime >= loopEnd){ vid.pause(); loopEnd = null; }
  const t = vid.currentTime;
  $('cursor').style.left = (100*t/ep.duration) + '%';
  const s = ep.subtasks.find(x=>t>=x.start_time && t<x.end_time) || ep.subtasks[ep.subtasks.length-1];
  $('caption').innerHTML = s ? `<b>#${s.id} [${s.action}]</b> ${s.subtask}` : '';
});
$('timeline').addEventListener('click', e=>{
  const r = e.currentTarget.getBoundingClientRect();
  vid.currentTime = DATA[cur].duration * (e.clientX - r.left) / r.width; loopEnd = null;
});
$('prev').onclick = ()=>{ if(cur>0) go(cur-1); };
$('next').onclick = ()=>{ if(cur<DATA.length-1) go(cur+1); };
$('export').onclick = ()=>{
  const blob = new Blob([JSON.stringify({version:1, exported_at:new Date().toISOString(), verdicts:store}, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'review_verdicts.json'; a.click();
};
renderMain();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = Path(cfg["paths"]["output_dir"]) / "review"
    media = out_dir / "media"
    media.mkdir(parents=True, exist_ok=True)

    episodes = select_episodes(cfg, args.category, args.n, args.seed)
    manifest = []
    for a in episodes:
        slug = a["episode_id"].replace("/", "__")
        src = data_dir / f"{a['episode_id']}.mp4"
        transcode(src, media / f"{slug}.mp4")
        manifest.append({
            "id": a["episode_id"], "task": a["task"], "file": f"{slug}.mp4",
            "duration": (a["total_frames"] - 1) / a["fps"],
            "flagged": bool(a["review_queue"]),
            "subtasks": [{k: s[k] for k in
                          ("id", "action", "hand", "object", "subtask",
                           "start_time", "end_time", "flags")}
                         for s in a["subtasks"]],
        })
        print(f"  {a['episode_id']}: {len(a['subtasks'])} subtasks "
              f"{'[flagged]' if a['review_queue'] else ''}")

    (out_dir / "index.html").write_text(
        HTML.replace("%%DATA%%", json.dumps(manifest)))
    n_flag = sum(m["flagged"] for m in manifest)
    print(f"\n{len(manifest)} episodes ({n_flag} flagged / "
          f"{len(manifest) - n_flag} clean) -> {out_dir / 'index.html'}")
    print("open it in a browser; export verdicts when done, then run "
          "report_verdicts.py on the downloaded JSON")


if __name__ == "__main__":
    main()
