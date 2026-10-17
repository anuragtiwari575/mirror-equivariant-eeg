r"""
run_all2.py  --  the last queue. Everything the paper still needs, in one run.

Start it once and leave it. About 52 hours. It waits for whatever is running now to finish, then works
through the list below, writing a log per part, and it remembers what finished in
corrected_results/queue_state2.json, so after a crash, a hang or a reboot the same command resumes at the
part that was interrupted instead of repeating what is already done.

    python run_all2.py --check     build every model this queue uses, on every montage, and stop.
                                   Run this FIRST. It takes a minute and catches a mistyped model name
                                   or a network that will not build before fifty hours are spent on it.
    python run_all2.py             the queue
    python run_all2.py --list      what the parts are and which are done
    python run_all2.py --from 4    start at part 4
    python run_all2.py --only 6,7  just these

What each part is for, and why it is in this order:

  1  CSP and FBCSP again, with oracle approximating shrinkage              ~3 h, processor only
     Required, not optional: the Methods section states that all three classical pipelines estimate
     covariance the same way, and that is only true once this has run. It also repairs FBCSP, which
     failed on four Cho subjects because a 4 Hz sub-band of a 64-channel montage can be rank deficient.

  2  the mirrored score of every test trial, BCI IV 2a and Cho             ~2.5 h
     Draws the paper's central claim instead of asserting it: s(x) against s(Mx), which lies exactly on
     the line s(Mx) = -s(x) for the construction and scatters around it for the augmentation.

  3  EEG-Conformer with the construction and with the augmentation         ~11 h
     The paper claims the construction wraps any backbone but shows it on two convolutional networks of
     the same family. A transformer is a different animal. If the gain survives there the claim is
     demonstrated; if it does not, that is worth knowing before a reviewer asks.

  4  learning curves on Cho and Dreyer                                     ~8 h
     The clinical argument about calibration time currently rests on two nine-subject datasets. This
     puts it on 52 and 87 subjects.

  5  Dreyer with three seeds                                              ~27 h
     Re-fitting the same model with the same seed moved a Dreyer mean by 0.75 accuracy points, which is
     the size of several differences reported on that dataset. Three seeds per subject makes those
     differences interpretable; the fits that already exist are not repeated.

  6  complexity table on an idle GPU                                       ~5 min
  7  every table, figure and number rebuilt                                ~2 min

Watch it from another window with:  python progress.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "corrected_results")
STATE = os.path.join(OUTDIR, "queue_state2.json")
STUDY = "step24_subject_specific.py"

ALL5 = "2a,2b,PhysioNet,Cho2017,Dreyer2023"
MAIN6 = "eegnet,shallow,eegnet_cr,shallow_cr,eegnet_mirror,shallow_mirror"
CONTROLS = "eegnet_rand,shallow_rand,eegnet_two,shallow_two"
CURVE4 = "eegnet,shallow,eegnet_mirror,shallow_mirror"

PARTS = [
    (1, "CSP and FBCSP again, with shrinkage", STUDY,
     ["--datasets", ALL5, "--models", "csp_lda,fbcsp", "--protocol", "session", "--redo"],
     "run2_classical.log", 3.0),

    # Cho2017 is out of parts 2, 4 and 5, and Cho2017 with PhysioNet out of part 3. The measured rate on
    # that corpus is about 1040 s for the six models of one subject -- ShallowConvNet alone costs 203 s
    # on its sixty-four channels against 43 s for EEGNet -- so the three parts together asked for about
    # 33 h of the queue's 64. What they would have bought is per-trial scores for a corpus whose
    # accuracy, asymmetry, contrasts and confirmatory test are already in the result file from the
    # earlier queue: the antisymmetry figure draws one corpus and BCI IV 2b is fully scored for it, so
    # what is given up is Cho's panel in the ROC figure and its two learning-curve points.
    (2, "mirrored scores for the antisymmetry figure, 2a", STUDY,
     ["--datasets", "2a", "--models", MAIN6, "--protocol", "session", "--scores", "--redo"],
     "run2_mirror_scores.log", 0.6),

    (3, "EEG-Conformer with the construction, the smallest corpus and the largest", STUDY,
     ["--datasets", "2a,Dreyer2023", "--models", "conformer_cr,conformer_mirror",
      "--protocol", "session"],
     "run2_conformer.log", 6.0),

    (4, "learning curve at a quarter of the training trials, Dreyer", STUDY,
     ["--datasets", "Dreyer2023", "--models", CURVE4, "--protocol", "session",
      "--train-fraction", "0.25"],
     "run2_curve25.log", 1.2),

    (5, "learning curve at a half of the training trials, Dreyer", STUDY,
     ["--datasets", "Dreyer2023", "--models", CURVE4, "--protocol", "session",
      "--train-fraction", "0.5"],
     "run2_curve50.log", 1.2),

    (6, "Dreyer with three seeds, every model that enters a contrast", STUDY,
     ["--datasets", "Dreyer2023", "--models", MAIN6 + "," + CONTROLS, "--protocol", "session",
      "--seeds", "0,1,2"],
     "run2_dreyer_seeds.log", 21.0),

    (7, "complexity table, on an idle GPU", "complexity.py", [], "run2_complexity.log", 0.1),

    (8, "every table, figure and number rebuilt", "make_paper_assets.py",
     ["--datasets", "all"], "run2_assets.log", 0.1),
]

# What --check builds, so that a wrong name costs a minute instead of a night.
CHECK_MODELS = sorted(set((MAIN6 + "," + CONTROLS + ",conformer_cr,conformer_mirror").split(",")))
CHECK_CHANNELS = [3, 22, 27, 64]


# ----------------------------------------------------------------- state
def load_state():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    os.makedirs(OUTDIR, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, sort_keys=True)
    os.replace(tmp, STATE)


def stamp():
    return f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"


def hhmm(seconds):
    seconds = int(max(0, seconds))
    return (f"{seconds // 3600}h {seconds % 3600 // 60:02d}m" if seconds >= 3600
            else f"{seconds // 60}m {seconds % 60:02d}s")


def say(line):
    print(line, flush=True)
    try:
        with open(os.path.join(HERE, "run_all2_queue.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def tail(path, n=1):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f.readlines() if l.strip()]
        return lines[-n:] if lines else []
    except Exception:
        return []


# ----------------------------------------------------------------- preflight
def check():
    """Build every model this queue uses, on every montage width, and report. Nothing is fitted."""
    say(f"  {stamp()}  building every model of this queue on every montage")
    try:
        sys.path.insert(0, HERE)
        import torch                                           # noqa: F401
        import step24_subject_specific as S24
    except Exception as e:
        say(f"  could not import the study script: {e!r}")
        return 1
    bad = 0
    for C in CHECK_CHANNELS:
        perm = list(range(C))[::-1]
        line = f"  {C:>3} channels: "
        for m in CHECK_MODELS:
            try:
                base = m[:-3] if m.endswith("_cr") else m
                mk = {"eegnet": lambda: S24.EEGNet(C, S24.CROP, 2),
                      "shallow": lambda: S24.Shallow(C, S24.CROP, 2),
                      "conformer": lambda: S24.Conformer(C, S24.CROP, 2),
                      "eegnet_mirror": lambda: S24.Mirror(S24.EEGNet(C, S24.CROP, 1), perm),
                      "shallow_mirror": lambda: S24.Mirror(S24.Shallow(C, S24.CROP, 1), perm),
                      "conformer_mirror": lambda: S24.Mirror(S24.Conformer(C, S24.CROP, 1), perm),
                      "eegnet_rand": lambda: S24.Mirror(S24.EEGNet(C, S24.CROP, 1),
                                                        S24.random_involution(C, 0)),
                      "shallow_rand": lambda: S24.Mirror(S24.Shallow(C, S24.CROP, 1),
                                                         S24.random_involution(C, 0)),
                      "eegnet_two": lambda: S24.TwoHeads(S24.EEGNet(C, S24.CROP, 1),
                                                         S24.EEGNet(C, S24.CROP, 1)),
                      "shallow_two": lambda: S24.TwoHeads(S24.Shallow(C, S24.CROP, 1),
                                                          S24.Shallow(C, S24.CROP, 1))}[base]
                net = mk()
                import torch as _t
                with _t.no_grad():
                    out = net(_t.zeros(2, C, S24.CROP))
                assert out.shape[0] == 2, f"{m}: unexpected output shape {tuple(out.shape)}"
                line += "."
            except Exception as e:
                bad += 1
                line += "X"
                say(f"  {m} on {C} channels FAILED: {e!r}")
        say(line + f"   ({len(CHECK_MODELS)} models)")
    for script in sorted({p[2] for p in PARTS}):
        say(f"  {'found   ' if os.path.exists(os.path.join(HERE, script)) else 'MISSING '}{script}")
    say("")
    say("  everything builds; the queue is safe to start" if not bad else
        f"  {bad} model/montage combinations did not build -- fix them before starting the queue")
    return 0 if not bad else 1


# ----------------------------------------------------------------- waiting
def fits_running():
    me = os.getpid()
    watch = ("step24_subject_specific", "complexity.py")
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | ? {$_.Name -eq 'python.exe'} |"
             " % { \"$($_.ProcessId)`t$($_.CommandLine)\" }"],
            capture_output=True, text=True, timeout=25)
        out = []
        for line in r.stdout.splitlines():
            if "\t" not in line:
                continue
            pid, cmd = line.split("\t", 1)
            try:
                if int(pid) == me:
                    continue
            except ValueError:
                continue
            if any(w in cmd for w in watch) and "run_all" not in cmd:
                out.append(cmd.strip()[:110])
        return out
    except Exception:
        return []


def wait_for_free(every=60):
    first, told, t0 = True, 0.0, time.time()
    while True:
        busy = fits_running()
        if not busy:
            if not first:
                say(f"  the machine is free again after {hhmm(time.time() - t0)}")
            return
        if first or time.time() - told > 300:
            say(f"  {stamp()}  waiting for a run that is already going:")
            for b in busy:
                say(f"      {b}")
            told = time.time()
        first = False
        time.sleep(every)


# ----------------------------------------------------------------- running
def run_part(num, title, script, args, log, est, state, total):
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        say(f"  {script} is not in this folder; part {num} skipped")
        state[str(num)] = {"status": "missing", "when": stamp()}
        save_state(state)
        return
    say("")
    say("=" * 78)
    say(f"  {num} of {total}  {title}")
    say(f"          started {stamp()}, expected about {est:.1f} h")
    say("=" * 78)
    cmd = [sys.executable, "-u", path] + args
    say("  " + " ".join(cmd[2:]))
    logpath = os.path.join(HERE, log)
    t0 = time.time()
    with open(logpath, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
        try:
            last = 0.0
            while proc.poll() is None:
                time.sleep(5)
                if time.time() - last >= 60:
                    last = time.time()
                    t = tail(logpath)
                    note = t[0][:96] if t else "(nothing written yet)"
                    say(f"  {stamp()}  part {num}, running {hhmm(time.time() - t0)} | {note}")
        except KeyboardInterrupt:
            say("\n  Ctrl+C: stopping this part. Every fit that finished is in the result file;")
            say("  start again with  python run_all2.py  and it resumes here.")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
            state[str(num)] = {"status": "interrupted", "seconds": round(time.time() - t0),
                               "when": stamp()}
            save_state(state)
            raise
    took = time.time() - t0
    ok = proc.returncode == 0
    state[str(num)] = {"status": "done" if ok else "failed", "code": proc.returncode,
                       "seconds": round(took), "when": stamp()}
    save_state(state)
    if ok:
        say(f"  part {num} finished in {hhmm(took)}")
    else:
        say(f"  part {num} stopped with code {proc.returncode} after {hhmm(took)}; see {log}")
        for t in tail(logpath, 6):
            say(f"      {t[:104]}")
        say("  carrying on with the next part")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="the last queue")
    ap.add_argument("--check", action="store_true", help="build every model and stop")
    ap.add_argument("--list", action="store_true", help="show the parts and stop")
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--only")
    ap.add_argument("--again")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.chdir(HERE)
    if args.check:
        return check()

    state = load_state()
    if args.again:
        for n in [x.strip() for x in args.again.split(",") if x.strip()]:
            state.pop(n, None)
        save_state(state)
        say(f"  forgotten: parts {args.again}")

    if args.only:
        want = [int(x) for x in args.only.split(",") if x.strip()]
        queue, skip_done = [p for n in want for p in PARTS if p[0] == n], False
    else:
        queue, skip_done = [p for p in PARTS if p[0] >= args.start], True

    if args.list:
        say(f"  folder: {HERE}")
        say(f"  {'part':<5}{'status':<13}{'took':>9}  what it is")
        for num, title, _s, _a, _l, est in PARTS:
            st = state.get(str(num), {})
            took = hhmm(st["seconds"]) if st.get("seconds") else ""
            say(f"  {num:<5}{st.get('status', '-'):<13}{took:>9}  {title}  (~{est:.1f} h)")
        return 0

    todo = [p for p in queue if not (skip_done and state.get(str(p[0]), {}).get("status") == "done")]
    done_already = [p[0] for p in queue if p not in todo]
    hours = sum(p[5] for p in todo)

    say("")
    say("=" * 78)
    say(f"  run_all2.py starting {stamp()}")
    say(f"  folder: {HERE}")
    if done_already:
        say(f"  already finished, skipping: parts {', '.join(str(n) for n in done_already)}")
    say(f"  to run: parts {', '.join(str(p[0]) for p in todo) or '(none)'}"
        f"  --  about {hours:.1f} h, so done by "
        f"{(datetime.datetime.now() + datetime.timedelta(hours=hours)):%a %d %b %H:%M}")
    say("=" * 78)

    if args.dry_run:
        for num, title, script, a, log, est in todo:
            say(f"  {num}: {script} {' '.join(a)}   > {log}")
        return 0
    if not todo:
        say("  nothing left to do.")
        return 0

    if not args.no_wait:
        wait_for_free()

    t0 = time.time()
    try:
        for num, title, script, a, log, est in todo:
            run_part(num, title, script, a, log, est, state, len(PARTS))
    except KeyboardInterrupt:
        return 1

    say("")
    say("=" * 78)
    say(f"  the queue finished at {stamp()}, {hhmm(time.time() - t0)} in all")
    for num, title, *_ in todo:
        st = state.get(str(num), {})
        say(f"    part {num}  {st.get('status', '?'):<12}{hhmm(st.get('seconds', 0)):>9}  {title}")
    say("")
    say(r"    corrected_results\paper_assets\summary.txt    the numbers")
    say(r"    corrected_results\paper_assets\MANIFEST.txt   what was built and what is missing")
    say("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
