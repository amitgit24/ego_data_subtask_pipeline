"""Human review UI builder for the single-pass subtask pipeline.

Sibling of ../../../VLM_seg_subtask_pipeline/tools/review/make_review.py and the
Kinematics reviewer. Same self-contained review page and verdict axes so all
three pipelines are judged by an IDENTICAL human protocol (the point of running
them in parallel is a fair head-to-head). Differs only in the source of
annotations (this pipeline's outputs/annotations) and a pipeline-specific
localStorage key so verdicts don't collide with the other reviewers in one
browser.

    python make_review.py --n 20 [--seed 11]
    -> VLM_subtask_pipeline/outputs/review/<task>/vlm_subtask_pipeline_review.html
       (folder = the task name when every selected episode is the same
       task, else 'mixed_tasks')
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
from common import load_config  # noqa: E402


def select_episodes(cfg, n, seed, episode_ids=None):
    ann_dir = Path(cfg["paths"]["output_dir"]) / "annotations"
    anns = [json.loads(f.read_text()) for f in sorted(ann_dir.glob("*.json"))]
    if not anns:
        raise SystemExit(f"no annotations in {ann_dir} — run the pipeline first")

    if episode_ids:
        wanted = set(episode_ids)
        picked = [a for a in anns if a["episode_id"] in wanted]
        missing = wanted - {a["episode_id"] for a in picked}
        if missing:
            raise SystemExit(f"no annotation found for: {sorted(missing)}")
        return picked

    rng = random.Random(seed)
    flagged = [a for a in anns if a["review_queue"]]
    clean = [a for a in anns if not a["review_queue"]]
    rng.shuffle(flagged)
    rng.shuffle(clean)

    picked, seen_tasks = [], set()
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


def review_folder_name(episodes):
    """outputs/review/<this> — named after what's actually being reviewed,
    so batches for different tasks never overwrite each other or pile up
    unlabeled in one flat folder."""
    tasks = {a["episode_id"].split("/")[0] for a in episodes}
    return next(iter(tasks)) if len(tasks) == 1 else "mixed_tasks"


def transcode(src, dst, force=False):
    if dst.exists() and not force:
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vf", "scale=1080:-2", "-c:v", "libx264", "-crf", "18",
         "-preset", "slow", "-an", "-movflags", "+faststart", str(dst)],
        check=True)


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>EgoDex single-pass subtask review</title>
<style>
 :root{--vw:960px;--bg:#15171c;--bg2:#1c1f26;--bg3:#242832;--fg:#e8e9ec;--fg2:#b7bac2;--fg3:#848993;--line:#333744;--accent:#6ea8ff}
 *{box-sizing:border-box}
 body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh;font-size:16px;background:var(--bg);color:var(--fg)}
 #side{width:280px;overflow-y:auto;border-right:1px solid var(--line);padding:8px;flex-shrink:0;background:var(--bg2)}
 #side h3{color:var(--fg)}
 #side .ep{padding:6px 8px;border-radius:5px;cursor:pointer;font-size:13px;margin:2px 0;color:var(--fg2)}
 #side .ep:hover{background:var(--bg3)}
 #side .ep.active{background:#2b3a5c;font-weight:600;color:var(--fg)}
 #side .ep .st{float:right;font-size:12px}
 #main{flex:1;overflow-y:auto;padding:16px 26px}
 video{width:var(--vw);max-width:100%;background:#000;border-radius:8px;border:1px solid var(--line)}
 #timeline{position:relative;width:var(--vw);max-width:100%;height:40px;margin:8px 0 4px;
   background:var(--bg3);border-radius:5px;overflow:hidden;cursor:pointer}
 #timeline .blk{position:absolute;top:0;height:100%;border-right:2px solid var(--bg);
   font-size:11px;color:#fff;overflow:hidden;padding:3px 4px;box-sizing:border-box}
 #timeline .cursor{position:absolute;top:0;width:2px;height:100%;background:#ff5555}
 #caption{width:var(--vw);max-width:100%;min-height:30px;font-size:17px;padding:8px 2px;color:var(--fg2)}
 #title{font-size:22px;color:var(--fg)}
 #task{font-size:19px;font-weight:700;margin:2px 0 10px;color:var(--fg)}
 .vbtn{border:1px solid #555b6b;background:var(--bg3);color:var(--fg2);border-radius:5px;padding:5px 14px;cursor:pointer;margin:0 3px;font-size:14px}
 .vbtn.good{background:#2e9e44;color:#fff;border-color:#2e9e44}
 .vbtn.bad{background:#d33;color:#fff;border-color:#d33}
 .play{cursor:pointer;color:var(--accent);font-weight:700;white-space:nowrap;font-size:16px}
 .flag{color:#e0a030;font-size:13px;font-weight:600}
 #epverdict{margin:12px 0;padding:14px 16px;background:var(--bg2);border:1px solid var(--line);border-radius:8px;width:var(--vw);max-width:100%;box-sizing:border-box;font-size:16px}
 input[type=text]{font-size:14px;padding:4px 7px;background:var(--bg3);color:var(--fg);border:1px solid var(--line);border-radius:4px}
 #topbar{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:5;border-bottom:1px solid var(--line)}
 button.big{padding:7px 16px;font-size:15px;border-radius:6px;border:1px solid #555b6b;background:var(--bg3);color:var(--fg);cursor:pointer}
 button.big:hover{background:#2b3a5c}
 .done{color:#4cc26a}.part{color:#e0a030}
 .hint{color:var(--fg3);font-size:13px}
 #seglist{width:var(--vw);max-width:100%}
 .segcard{border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:16px 0;background:var(--bg2)}
 .segcard.active{background:#20283d;border-color:var(--accent)}
 .segcard .sh{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:8px}
 .segcard .segtitle{font-size:19px;font-weight:700;color:var(--fg)}
 .segcard .segtime{font-size:15px;color:var(--fg3)}
 .segrow{font-size:16px;margin:6px 0;display:flex;gap:8px}
 .segrow .k{min-width:80px;color:var(--fg3);flex-shrink:0}
 .segrow .v{font-weight:600;color:var(--fg)}
 .sentence{font-size:17px;margin:10px 0;padding:10px 12px;background:var(--bg3);border-radius:6px;line-height:1.4;color:var(--fg)}
 .verdicts{display:flex;flex-direction:column;gap:8px;margin-top:10px;font-size:15px}
 .vgroup{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
 .vgroup .k{min-width:80px;color:var(--fg3)}
 .notein{width:280px;max-width:60%}
 .badge{font-size:12px;color:var(--fg3);border:1px solid var(--line);border-radius:4px;padding:1px 6px;margin-left:8px}
</style></head><body>
<div id="side"><h3 style="margin:4px">Episodes</h3><div id="eplist"></div></div>
<div id="main">
 <div id="topbar">
  <button class="big" id="prev">&larr; prev</button>
  <button class="big" id="next">next &rarr;</button>
  <button class="big" id="export">Export verdicts JSON</button>
  <span class="badge">pipeline: single-pass subtask</span>
  <span class="hint" id="progress"></span>
 </div>
 <h2 id="title"></h2><div id="task"></div>
 <video id="vid" controls preload="metadata"></video>
 <div id="timeline"></div><div id="caption"></div>
 <div id="epverdict"></div>
 <div id="seglist"></div>
 <p class="hint">Per segment: <b>&#9654; play</b> plays the segment with 0.5 s
 pre/post-roll (judge the cut points). <b>Boundary</b> = start/end cuts correct,
 nothing missed inside. <b>Label</b> = action+hand+object correct.
 <b>Sentence</b> = description factually right. Verdicts save automatically
 (localStorage); export when done.</p>
</div>
<script>
const DATA = %%DATA%%;
const LS = 'egodex_vlm_subtask_review_v1';
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
  return `<button class="vbtn ${state===want?(want?'good':'bad'):''}" onclick="${cb}">${label}</button>`;
}
function renderMain(){
  const ep = DATA[cur];
  $('title').textContent = ep.id;
  $('task').textContent = ep.task;
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
  $('seglist').innerHTML = ep.subtasks.map(s=>{
      const g = sv(ep, s.id);
      return `<div class="segcard" id="row${s.id}">
       <div class="sh">
         <span class="segtitle play" onclick="playSeg(${s.id})">&#9654; #${s.id} &mdash; ${s.action}</span>
         <span class="segtime">${s.start_time.toFixed(1)}&ndash;${s.end_time.toFixed(1)}s</span>
       </div>
       ${s.flags.length?`<div class="flag">flags: ${s.flags.join(', ')}</div>`:''}
       <div class="segrow"><span class="k">hand</span><span class="v">${s.hand}</span></div>
       <div class="segrow"><span class="k">object</span><span class="v">${s.object}</span></div>
       <div class="sentence">${s.subtask}</div>
       <div class="verdicts">
         <div class="vgroup"><span class="k">boundary</span>${btn('',g.boundary,true,'&#10003; correct',`segv(${s.id},'boundary',true)`)}${btn('',g.boundary,false,'&#10007; wrong',`segv(${s.id},'boundary',false)`)}</div>
         <div class="vgroup"><span class="k">label</span>${btn('',g.label,true,'&#10003; correct',`segv(${s.id},'label',true)`)}${btn('',g.label,false,'&#10007; wrong',`segv(${s.id},'label',false)`)}</div>
         <div class="vgroup"><span class="k">sentence</span>${btn('',g.sentence,true,'&#10003; correct',`segv(${s.id},'sentence',true)`)}${btn('',g.sentence,false,'&#10007; wrong',`segv(${s.id},'sentence',false)`)}</div>
         <div class="vgroup"><span class="k">note</span><input class="notein" type="text" value="${(g.note||'').replace(/"/g,'&quot;')}" onchange="segnote(${s.id},this.value)"></div>
       </div>
      </div>`;}).join('');
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
  document.querySelectorAll('.segcard').forEach(r=>r.classList.remove('active'));
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
  const blob = new Blob([JSON.stringify({version:1, pipeline:'vlm_subtask', exported_at:new Date().toISOString(), verdicts:store}, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'vlm_subtask_review_verdicts.json'; a.click();
};
renderMain();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="exact episode ids (task/idx) to review, "
                         "overriding random --n selection")
    ap.add_argument("--force-transcode", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])

    episodes = select_episodes(cfg, args.n, args.seed, episode_ids=args.episodes)
    out_dir = (Path(cfg["paths"]["output_dir"]) / "review"
              / review_folder_name(episodes))
    media = out_dir / "media"
    media.mkdir(parents=True, exist_ok=True)

    manifest = []
    for a in episodes:
        slug = a["episode_id"].replace("/", "__")
        src = data_dir / f"{a['episode_id']}.mp4"
        transcode(src, media / f"{slug}.mp4", force=args.force_transcode)
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

    (out_dir / "vlm_subtask_pipeline_review.html").write_text(
        HTML.replace("%%DATA%%", json.dumps(manifest)))
    n_flag = sum(m["flagged"] for m in manifest)
    print(f"\n{len(manifest)} episodes ({n_flag} flagged / "
          f"{len(manifest) - n_flag} clean) -> {out_dir / 'vlm_subtask_pipeline_review.html'}")
    print("open it in a browser; export verdicts when done, then run "
          "report_verdicts.py on the downloaded JSON")


if __name__ == "__main__":
    main()
