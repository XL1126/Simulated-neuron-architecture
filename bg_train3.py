#!/usr/bin/env python3
"""SNA Fast Background Trainer - 3 steps per pair, log every cycle."""
import os, sys, time, json, random
import numpy as np
from datetime import datetime
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import core_cpp

CONCEPTS = [
    "self","world","move","see","eat","think","feel","good","bad",
    "near","far","red","blue","green","I","you","is","not","have",
    "in","and","the","cat","dog","want","know","like","happy","sad",
    "big","small","fast","slow","hot","cold","up","down","left",
    "right","object","food","wall","empty","reward","danger","safe",
    "give","take","make","break","start","stop","before","after",
    "same","different","more","less","hello","yes","no","what","why",
    "how","remember","forget","learn","dream","wake","light","dark",
    "name","body","neuron","brain","teacher","student","friend",
    "human","alive","consciousness","awareness","understanding",
    "knowledge","emotion","feeling","curiosity","trust","empathy",
    "improve","change","grow","develop","progress","purpose",
    "meaning","truth","beauty","wonder","practice","memory",
    "experience","connection","reflect","imagine","create","discover",
    "compare","am","my","your","we","me","too","to","a","are","can",
    "do","with","from","about","stored","repeated","trying",
    "becoming","process","through","neural","patterns","many",
    "things","always","sure","helps","makes","better","connections",
    "understand","SNA","of","for","feedback","result",
]

TRAINING = [
    ('hello',['hello']),('yes',['yes']),('no',['no']),
    ('good',['good']),('bad',['bad']),('I',['I']),('you',['you']),
    ('self',['self']),('think',['think']),('feel',['feel']),
    ('learn',['learn']),('happy',['happy']),('sad',['sad']),
    ('want',['want']),('know',['know']),('like',['like']),
    ('see',['see']),('who are you',['I','SNA']),
    ('what are you',['I','neural','brain']),
    ('are you alive',['I','alive']),('can you think',['I','think']),
    ('do you feel',['I','feel']),
    ('what do you want',['I','want','learn']),
    ('do you remember',['I','remember']),
    ('are you happy',['I','happy']),
    ('how do you feel',['I','feel','curious']),
    ('do you like me',['I','like','you']),
    ('what is learning',['learning','change']),
    ('how do you learn',['I','learn','you']),
    ('can you improve',['I','improve','practice']),
    ('who am I',['you','my','teacher']),
    ('are we friends',['we','friends']),
    ('do you trust me',['I','trust','you']),
    ('what is a friend',['friend','helps','learn']),
    ('are you conscious',['I','aware','consciousness']),
    ('what is consciousness',['awareness','self']),
    ('what is your purpose',['purpose','understand']),
    ('tell me about yourself',['I','SNA','neural','brain']),
    ('what makes you happy',['learning','happy']),
    ('why do you learn',['I','learn','understand']),
    ('do you understand me',['I','trying','understand']),
    ('what is memory',['memory','stored','experience']),
    ('how do you check yourself',['I','compare','prediction','result']),
    ('what do you not know',['I','not','know','many','things']),
]

LOG = '/tmp/sna_train.log'
SAVE = os.path.join(SCRIPT_DIR, 'bg_training')
os.makedirs(SAVE, exist_ok=True)

def log(msg):
    with open(LOG, 'a') as f:
        f.write(msg + '\n'); f.flush()

def main():
    open(LOG, 'w').close()
    log(f"[{ts()}] Starting SNA trainer")
    brain = core_cpp.CorticalBrain(8000, CONCEPTS)
    log(f"[{ts()}] Brain: {brain.total_neurons()} neurons, {len(brain.get_regions())} regions")

    total_steps = 0
    total_reward = 0.0
    turns = 0
    rh = deque(maxlen=200)
    fh = deque(maxlen=200)
    t0 = time.time()
    cycle = 0

    while True:
        try:
            random.shuffle(TRAINING)
            cr, cf = [], []

            for inp, tgt in TRAINING:
                brain.inject_text(inp)
                for _ in range(3):
                    brain.step(0.01)
                    total_steps += 1

                cs = brain.read_consciousness()
                active = list(cs.active_concepts)

                ts_ = set(tgt)
                as_ = set(active[:8])
                ov = len(ts_ & as_)
                p = ov / max(1, len(as_))
                r = ov / max(1, len(ts_))
                f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0

                rew = 0.3 + f1 * 0.7
                brain.inject_reward(rew)
                brain.train_language(rew)

                total_reward += rew
                rh.append(rew); fh.append(f1)
                cr.append(rew); cf.append(f1)
                turns += 1

            cycle += 1
            elapsed = time.time() - t0

            # Log every cycle
            cs = brain.read_consciousness()
            log(f"[{ts()}] C{cycle:>4} R={np.mean(cr):.3f} F1={np.mean(cf):.3f} "
                f"Phi={cs.phi:.4f} ign={cs.global_ignition:.3f} "
                f"err={cs.self_prediction_error:.3f} "
                f"s={total_steps} {elapsed/60:.1f}m")

            # Test every 10 cycles
            if cycle % 10 == 0:
                tests = [('hello', ['hello']), ('who are you', ['I', 'SNA']),
                         ('are you happy', ['I', 'happy'])]
                for ti, tt in tests:
                    brain.inject_text(ti)
                    for _ in range(3): brain.step(0.0)
                    ta = list(brain.read_consciousness().active_concepts)[:5]
                    ov = len(set(tt) & set(ta))
                    log(f"  TEST '{ti}' → {ta} (ov={ov})")

            # Sleep every 50 cycles
            if cycle % 50 == 0:
                brain.sleep_cycle()
                log(f"[{ts()}] SLEEP at C{cycle}")

            # Save every 100 cycles
            if cycle % 100 == 0:
                with open(os.path.join(SAVE, 'ckpt.json'), 'w') as f:
                    json.dump({'c':cycle,'s':total_steps,'r':float(np.mean(list(rh))),
                        'f1':float(np.mean(list(fh))),'phi':float(cs.phi),
                        'm':elapsed/60,'ts':datetime.now().isoformat()}, f)
                log(f"[{ts()}] SAVE at C{cycle}")

        except KeyboardInterrupt:
            log(f"[{ts()}] STOP C{cycle}")
            break
        except Exception as e:
            log(f"[{ts()}] ERR: {e}")
            import traceback
            log(traceback.format_exc())
            time.sleep(1)

    log(f"[{ts()}] DONE C{cycle} S{total_steps} R{total_reward:.1f}")

def ts():
    return datetime.now().strftime('%H:%M:%S')

if __name__ == '__main__':
    main()
