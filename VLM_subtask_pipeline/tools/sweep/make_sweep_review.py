"""Blind human A/B review page for the window-geometry sweep.

For each episode, shows the video once plus one segmentation TIMELINE PER
CONFIG, anonymized as A/B/C/... and SHUFFLED PER EPISODE (deterministic per
episode id + seed) so the reviewer cannot learn which letter is which config.
The reviewer picks the best and worst timeline per episode (clicking a
timeline's blocks seeks/plays the video with that config's captions, so each
candidate segmentation can be experienced before judging).

Export downloads a verdicts JSON; score it with:
    python report_sweep_verdicts.py sweep_review_verdicts.json
which un-blinds the letters and tallies best/worst counts per config.

    python make_sweep_review.py [--seed 7]
    -> outputs/sweep/review/index.html
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "pipeline"))
from common import load_config  # noqa: E402


def transcode(src, dst, force=False):
    if dst.exists() and not force:
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vf", "scale=1080:-2", "-c:v", "libx264", "-crf", "18",
         "-preset", "slow", "-an", "-movflags", "+faststart", str(dst)],
        check=True)


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Sweep A/B review — window geometry</title>
<style>
 :root{--vw:960px;--bg:#15171c;--bg2:#1c1f26;--bg3:#242832;--fg:#e8e9ec;--fg2:#b7bac2;--fg3:#848993;--line:#333744;--accent:#6ea8ff}
 *{box-sizing:border-box}
 body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh;font-size:16px;background:var(--bg);color:var(--fg)}
 #side{width:280px;overflow-y:auto;border-right:1px solid var(--line);padding:8px;flex-shrink:0;background:var(--bg2)}
 #side .ep{padding:6px 8px;border-radius:5px;cursor:pointer;font-size:13px;margin:2px 0;color:var(--fg2)}
 #side .ep:hover{background:var(--bg3)}
 #side .ep.active{background:#2b3a5c;font-weight:600;color:var(--fg)}
 #side .ep .st{float:right;font-size:12px}
 #main{flex:1;overflow-y:auto;padding:16px 26px}
 video{width:var(--vw);max-width:100%;background:#000;border-radius:8px;border:1px solid var(--line)}
 #title{font-size:22px}
 #task{font-size:18px;font-weight:700;margin:2px 0 10px}
 #caption{width:var(--vw);max-width:100%;min-height:30px;font-size:16px;padding:6px 2px;color:var(--fg2)}
 .cand{width:var(--vw);max-width:100%;margin:14px 0;padding:12px 14px;background:var(--bg2);border:1px solid var(--line);border-radius:10px}
 .cand.activecand{border-color:var(--accent);background:#20283d}
 .cand .hd{display:flex;align-items:center;gap:12px;margin-bottom:8px}
 .cand .letter{font-size:20px;font-weight:800}
 .tl{position:relative;width:100%;height:36px;background:var(--bg3);border-radius:5px;overflow:hidden;cursor:pointer}
 .tl .blk{position:absolute;top:0;height:100%;border-right:2px solid var(--bg);font-size:11px;color:#fff;overflow:hidden;padding:2px 4px}
 .tl .cursor{position:absolute;top:0;width:2px;height:100%;background:#ff5555}
 .vbtn{border:1px solid #555b6b;background:var(--bg3);color:var(--fg2);border-radius:5px;padding:5px 14px;cursor:pointer;font-size:14px}
 .vbtn.best{background:#2e9e44;color:#fff;border-color:#2e9e44}
 .vbtn.worst{background:#d33;color:#fff;border-color:#d33}
 #topbar{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:5;border-bottom:1px solid var(--line)}
 button.big{padding:7px 16px;font-size:15px;border-radius:6px;border:1px solid #555b6b;background:var(--bg3);color:var(--fg);cursor:pointer}
 button.big:hover{background:#2b3a5c}
 .hint{color:var(--fg3);font-size:13px}
 .done{color:#4cc26a}
 input[type=text]{font-size:14px;padding:4px 7px;background:var(--bg3);color:var(--fg);border:1px solid var(--line);border-radius:4px;width:340px}
</style></head><body>
<div id="side"><h3 style="margin:4px">Episodes</h3><div id="eplist"></div></div>
<div id="main">
 <div id="topbar">
  <button class="big" id="prev">&larr; prev</button>
  <button class="big" id="next">next &rarr;</button>
  <button class="big" id="export">Export verdicts JSON</button>
  <span class="hint" id="progress"></span>
 </div>
 <h2 id="title"></h2><div id="task"></div>
 <video id="vid" controls preload="metadata"></video>
 <div id="caption"></div>
 <div id="cands"></div>
 <p class="hint">Each lettered card is the SAME video segmented by a different
 (hidden) window configuration. Click a timeline block to play that segment
 with that candidate's captions. Then mark the <b>best</b> and <b>worst</b>
 segmentation of the episode. Letters are shuffled per episode.</p>
</div>
<script>
const DATA = %%DATA%%;
const LS = 'egodex_sweep_review_v1';
let store = JSON.parse(localStorage.getItem(LS) || '{}');
let cur = 0, loopEnd = null, activeCand = 0;

const vid = document.getElementById('vid');
const $ = id => document.getElementById(id);

function v(ep){ if(!store[ep.id]) store[ep.id] = {best:null, worst:null, note:''}; return store[ep.id]; }
function save(){ localStorage.setItem(LS, JSON.stringify(store)); renderSide(); renderProgress(); }
function epDone(ep){ const e = store[ep.id]; return e && e.best!==null && e.worst!==null; }

function renderSide(){
  $('eplist').innerHTML = DATA.map((ep,i)=>
    `<div class="ep ${i===cur?'active':''}" onclick="go(${i})">${ep.id}
     <span class="st done">${epDone(ep)?'&#10003;':''}</span></div>`).join('');
}
function renderProgress(){
  $('progress').textContent = ` judged ${DATA.filter(epDone).length}/${DATA.length} episodes`;
}
function renderMain(){
  const ep = DATA[cur];
  $('title').textContent = ep.id;
  $('task').textContent = ep.task;
  vid.src = 'media/' + ep.file; loopEnd = null; activeCand = 0;
  const e = v(ep);
  $('cands').innerHTML = ep.cands.map((c,ci)=>{
    const tl = c.subtasks.map((s,si)=>
      `<div class="blk" style="left:${100*s.start_time/ep.duration}%;width:${100*(s.end_time-s.start_time)/ep.duration}%;
        background:hsl(${(si*67)%360},55%,48%)" onclick="playSeg(${ci},${si});event.stopPropagation()">${s.action}</div>`).join('');
    return `<div class="cand ${ci===activeCand?'activecand':''}" id="cand${ci}">
      <div class="hd"><span class="letter">${c.letter}</span>
        <span class="hint">${c.subtasks.length} subtasks</span>
        <button class="vbtn ${e.best===c.letter?'best':''}" onclick="mark('best','${c.letter}')">best</button>
        <button class="vbtn ${e.worst===c.letter?'worst':''}" onclick="mark('worst','${c.letter}')">worst</button>
      </div>
      <div class="tl" id="tl${ci}">${tl}<div class="cursor" id="cursor${ci}"></div></div>
    </div>`;}).join('') +
    `<div style="margin:10px 0">note: <input type="text" value="${(e.note||'').replace(/"/g,'&quot;')}" onchange="epnote(this.value)"></div>`;
  renderSide(); renderProgress();
}
window.go = i => { cur = i; renderMain(); };
window.mark = (field, letter) => {
  const e = v(DATA[cur]);
  e[field] = (e[field]===letter)?null:letter;
  if(field==='best' && e.worst===letter) e.worst=null;
  if(field==='worst' && e.best===letter) e.best=null;
  save(); renderMain();
};
window.epnote = val => { v(DATA[cur]).note = val; save(); };
window.playSeg = (ci, si) => {
  activeCand = ci;
  const s = DATA[cur].cands[ci].subtasks[si];
  vid.currentTime = Math.max(0, s.start_time - 0.5);
  loopEnd = Math.min(DATA[cur].duration, s.end_time + 0.5);
  vid.play();
  document.querySelectorAll('.cand').forEach(x=>x.classList.remove('activecand'));
  $('cand'+ci).classList.add('activecand');
};
vid.addEventListener('timeupdate', ()=>{
  const ep = DATA[cur];
  if(loopEnd !== null && vid.currentTime >= loopEnd){ vid.pause(); loopEnd = null; }
  const t = vid.currentTime;
  ep.cands.forEach((c,ci)=>{ const cu=$('cursor'+ci); if(cu) cu.style.left = (100*t/ep.duration)+'%'; });
  const c = ep.cands[activeCand];
  const s = c.subtasks.find(x=>t>=x.start_time && t<x.end_time) || c.subtasks[c.subtasks.length-1];
  $('caption').innerHTML = s ? `<b>[${c.letter}] ${s.action}</b> — ${s.subtask}` : '';
});
document.addEventListener('click', e=>{
  const tl = e.target.closest('.tl');
  if(tl && e.target===tl){
    const ci = parseInt(tl.id.slice(2));
    const r = tl.getBoundingClientRect();
    activeCand = ci;
    vid.currentTime = DATA[cur].duration*(e.clientX-r.left)/r.width; loopEnd=null;
  }
});
$('prev').onclick = ()=>{ if(cur>0) go(cur-1); };
$('next').onclick = ()=>{ if(cur<DATA.length-1) go(cur+1); };
$('export').onclick = ()=>{
  const blob = new Blob([JSON.stringify({version:1, kind:'sweep_ab', exported_at:new Date().toISOString(), verdicts:store}, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download='sweep_review_verdicts.json'; a.click();
};
renderMain();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=7,
                    help="per-episode letter shuffle seed (also stored in the "
                         "blinding key so verdicts can be un-blinded)")
    ap.add_argument("--force-transcode", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    sweep_root = Path(cfg["paths"]["output_dir"]) / "sweep"
    manifest = json.loads((sweep_root / "sweep_manifest.json").read_text())
    configs = list(manifest["configs"])
    out_dir = sweep_root / "review"
    media = out_dir / "media"
    media.mkdir(parents=True, exist_ok=True)

    letters = [chr(ord("A") + i) for i in range(len(configs))]
    blinding = {}   # episode -> {letter: config}
    pages = []
    for e in manifest["episodes"]:
        slug = e.replace("/", "__")
        anns = {}
        for c in configs:
            f = sweep_root / c / "annotations" / f"{slug}.json"
            if f.exists():
                anns[c] = json.loads(f.read_text())
        if len(anns) != len(configs):
            print(f"skip {e}: only {len(anns)}/{len(configs)} configs done")
            continue
        transcode(data_dir / f"{e}.mp4", media / f"{slug}.mp4",
                  force=args.force_transcode)
        order = configs[:]
        random.Random(f"{args.seed}:{e}").shuffle(order)
        blinding[e] = dict(zip(letters, order))
        any_ann = anns[order[0]]
        pages.append({
            "id": e, "task": any_ann["task"], "file": f"{slug}.mp4",
            "duration": (any_ann["total_frames"] - 1) / any_ann["fps"],
            "cands": [{
                "letter": letter,
                "subtasks": [{k: s[k] for k in
                              ("id", "action", "subtask", "start_time", "end_time")}
                             for s in anns[order[i]]["subtasks"]],
            } for i, letter in enumerate(letters)],
        })

    (out_dir / "index.html").write_text(HTML.replace("%%DATA%%", json.dumps(pages)))
    (out_dir / "blinding_key.json").write_text(json.dumps(
        {"seed": args.seed, "map": blinding}, indent=2))
    print(f"{len(pages)} episodes x {len(configs)} blinded candidates "
          f"-> {out_dir / 'index.html'}")
    print(f"blinding key (do NOT open while reviewing) -> "
          f"{out_dir / 'blinding_key.json'}")


if __name__ == "__main__":
    main()
