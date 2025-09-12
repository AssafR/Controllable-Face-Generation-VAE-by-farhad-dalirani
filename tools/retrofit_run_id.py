#!/usr/bin/env python3
"""
Retrofit run_id into existing logs under logs/ for runs that predate run_id support.

Strategy:
- experiments_log.jsonl: ensure every entry has run_id. For missing ones, synthesize
  run_id = f"{config_name or 'unified'}_<YYYYMMDD_HHMMSS>" based on the entry timestamp.
- Build run windows from experiments_log.jsonl (start/end per run_id) to map analysis entries.
- detailed/loss analysis files: add run_id to entries whose timestamps fall within a
  known run window. If no window matches, synthesize run_id from entry timestamp.

Creates .bak backups before overwriting.
"""

import os
import json
from datetime import datetime
from glob import glob

LOG_DIR = os.path.join(os.getcwd(), 'logs')


def parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace('Z', ''))
    except Exception:
        return None


def synth_run_id(config_name: str, ts: str) -> str:
    base = config_name or 'unified'
    try:
        dt = parse_iso(ts)
        stamp = dt.strftime('%Y%m%d_%H%M%S') if dt else ts.replace(':', '').replace('-', '').replace('T', '_')[:15]
    except Exception:
        stamp = 'unknown'
    return f"{base}_{stamp}"


def load_experiments(path: str):
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries


def save_experiments(path: str, entries):
    bak = path + '.bak'
    try:
        if os.path.exists(path):
            os.replace(path, bak)
    except Exception:
        pass
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')


def build_run_windows(entries):
    """Return list of windows: {run_id, start: dt, end: dt or None}.
    If multiple run_start exist with same run_id, create separate windows.
    """
    windows = []
    # Fill missing run_id in memory first to build windows
    for e in entries:
        if 'run_id' not in e or not e['run_id']:
            e['run_id'] = synth_run_id(e.get('config_name'), e.get('timestamp', ''))
    # Pair starts and ends chronologically
    sorted_entries = sorted(entries, key=lambda x: x.get('timestamp', ''))
    active = {}
    for e in sorted_entries:
        ts = parse_iso(e.get('timestamp', ''))
        ev = e.get('event')
        rid = e.get('run_id')
        if ev == 'run_start':
            # start a new window instance
            windows.append({'run_id': rid, 'start': ts, 'end': None})
        elif ev == 'run_end':
            # match the last open window for this rid
            for w in reversed(windows):
                if w['run_id'] == rid and w['end'] is None:
                    w['end'] = ts
                    break
    return windows


def retrofit_experiments(exp_path: str):
    entries = load_experiments(exp_path)
    if not entries:
        return [], entries
    changed = False
    for e in entries:
        if 'run_id' not in e or not e['run_id']:
            e['run_id'] = synth_run_id(e.get('config_name'), e.get('timestamp', ''))
            changed = True
    if changed:
        save_experiments(exp_path, entries)
    return build_run_windows(entries), entries


def retrofit_analysis_file(path: str, windows):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return False
    if not isinstance(data, list):
        return False
    changed = False
    for item in data:
        if isinstance(item, dict) and (('run_id' not in item) or (not item.get('run_id'))):
            ts = parse_iso(item.get('timestamp', ''))
            assigned = False
            if ts:
                for w in windows:
                    start = w.get('start')
                    end = w.get('end')
                    if start and ((end and start <= ts <= end) or (end is None and ts >= start)):
                        item['run_id'] = w['run_id']
                        assigned = True
                        changed = True
                        break
            if not assigned:
                item['run_id'] = synth_run_id(item.get('config_name'), item.get('timestamp', ''))
                changed = True
    if changed:
        bak = path + '.bak'
        try:
            if os.path.exists(path):
                os.replace(path, bak)
        except Exception:
            pass
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    return changed


def main():
    exp_path = os.path.join(LOG_DIR, 'experiments_log.jsonl')
    windows, _ = retrofit_experiments(exp_path)
    # Patch analysis logs
    candidates = []
    candidates.extend(glob(os.path.join(LOG_DIR, '*loss_analysis*.json')))
    # Common names
    for name in ['loss_analysis.json', 'detailed_loss_analysis.json', 'research_loss_analysis.json']:
        p = os.path.join(LOG_DIR, name)
        if p not in candidates:
            candidates.append(p)
    changed_any = False
    for path in candidates:
        if os.path.exists(path):
            changed = retrofit_analysis_file(path, windows)
            changed_any = changed_any or changed
    print("Done. Updated analysis logs:" , changed_any, "| windows:", len(windows))

    # Patch image filenames retroactively by adding run_id via sidecar mapping log
    # We won't rename files, but we will add image index entries to experiments log
    try:
        # Collect images in sample_images/
        sample_dir = os.path.join(os.getcwd(), 'sample_images')
        if os.path.isdir(sample_dir):
            images = sorted(glob(os.path.join(sample_dir, '*.png')))
        else:
            images = []
        # Map by timestamp in filename if present falls within window; as fallback, just log without run_id changes
        for img in images:
            # Best-effort: if any window is open-ended, associate all images newer than start
            ts = None
            assigned = False
            for w in windows:
                if w.get('start') and (w.get('end') is None):
                    # Can't parse file mtime reliably across systems; just link by open window
                    # Log association entry
                    assoc = {
                        'event': 'image_index',
                        'path': img,
                        'run_id': w['run_id'],
                    }
                    # Append without rewriting whole file
                    with open(exp_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(assoc, ensure_ascii=False) + '\n')
                    assigned = True
                    break
            if not assigned:
                # If we can't assign, write an index without run_id
                assoc = {
                    'event': 'image_index',
                    'path': img,
                }
                with open(exp_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(assoc, ensure_ascii=False) + '\n')
        # Retro-rename old sample image files to new scheme where feasible
        # Patterns to migrate:
        #   <config>[_<run_id>]_generated_epoch_XXX[<suffix>].png -> <config>[_<run_id>]_epoch_XXX_generated[<suffix>].png
        #   <config>[_<run_id>]_reconstruction_epoch_XXX[<suffix>].png -> <config>[_<run_id>]_epoch_XXX_reconstruction[<suffix>].png
        import re
        migrate_patterns = [
            (re.compile(r"^(.*?)(?:_(\w+_\d{8}_\d{6}))?_generated_epoch_(\d{3})(.*)\.png$"),
             lambda m: f"{m.group(1)}_{m.group(2)}_epoch_{m.group(3)}_generated{m.group(4)}.png" if m.group(2) else f"{m.group(1)}_epoch_{m.group(3)}_generated{m.group(4)}.png"),
            (re.compile(r"^(.*?)(?:_(\w+_\d{8}_\d{6}))?_reconstruction_epoch_(\d{3})(.*)\.png$"),
             lambda m: f"{m.group(1)}_{m.group(2)}_epoch_{m.group(3)}_reconstruction{m.group(4)}.png" if m.group(2) else f"{m.group(1)}_epoch_{m.group(3)}_reconstruction{m.group(4)}.png"),
        ]
        for img in images:
            base = os.path.basename(img)
            dirn = os.path.dirname(img)
            for pat, repl in migrate_patterns:
                mm = pat.match(base)
                if mm:
                    new_name = repl(mm)
                    new_path = os.path.join(dirn, new_name)
                    if new_path != img and not os.path.exists(new_path):
                        try:
                            os.rename(img, new_path)
                            # Log rename
                            with open(exp_path, 'a', encoding='utf-8') as f:
                                f.write(json.dumps({'event': 'image_renamed', 'from': img, 'to': new_path}, ensure_ascii=False) + '\n')
                        except Exception:
                            pass
                    break
    except Exception:
        pass

    # Retro-rename checkpoints to include run_id for easier resume-by-id
    try:
        import torch, re
        ckpt_dir = os.path.join(os.getcwd(), 'checkpoints')
        all_pths = glob(os.path.join(ckpt_dir, '*.pth'))
        pat_training = re.compile(r'^(.*)_training_checkpoint\.pth$')
        pat_epoch = re.compile(r'^(.*)_checkpoint_epoch_(\d{3})\.pth$')
        for ckpt in all_pths:
            base = os.path.basename(ckpt)
            rid = None
            cfg_name = None
            try:
                chk = torch.load(ckpt, map_location='cpu')
                cfg = chk.get('config', {}) if isinstance(chk, dict) else {}
                rid = cfg.get('run_id')
                cfg_name = cfg.get('config_name')
            except Exception:
                pass
            # If no run_id is discoverable, try infer by windows and mtime
            if not rid:
                mtime = datetime.fromtimestamp(os.path.getmtime(ckpt))
                # if we know cfg_name prefer that, else use prefix from filename later
                candidates = windows
                if cfg_name:
                    candidates = [w for w in windows if w.get('run_id', '').startswith(cfg_name)]
                # find covering window
                chosen = None
                for w in candidates:
                    st, en = w.get('start'), w.get('end')
                    if st and ((en and st <= mtime <= en) or (en is None and mtime >= st)):
                        chosen = w
                        break
                if not chosen and candidates:
                    chosen = candidates[-1]
                if chosen:
                    rid = chosen['run_id']

            # Determine new filename if pattern matches
            new_name = None
            m = pat_training.match(base)
            if m and rid:
                prefix = m.group(1)
                # Avoid double-inserting run_id
                if rid not in base:
                    new_name = f"{prefix}_{rid}_training_checkpoint.pth"
            else:
                m2 = pat_epoch.match(base)
                if m2 and rid:
                    prefix = m2.group(1)
                    epoch = m2.group(2)
                    if rid not in base:
                        new_name = f"{prefix}_{rid}_checkpoint_epoch_{epoch}.pth"

            if new_name:
                new_path = os.path.join(ckpt_dir, new_name)
                if new_path != ckpt and not os.path.exists(new_path):
                    try:
                        os.rename(ckpt, new_path)
                        with open(exp_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps({'event': 'checkpoint_renamed', 'from': ckpt, 'to': new_path}, ensure_ascii=False) + '\n')
                    except Exception:
                        pass
    except Exception:
        pass


if __name__ == '__main__':
    main()


