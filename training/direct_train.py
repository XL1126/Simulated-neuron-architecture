#!/usr/bin/env python3
"""
SNA Direct Concept Trainer v7
直接概念训练 — 利用新加入的 learnable concept weights

关键改进：直接调用 train_concept() 强化概念映射
"""
import os, sys, time, json, random, signal
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
    ('see',['see']),
    ('who are you',['I','SNA']),('what are you',['I','neural','brain']),
    ('are you alive',['I','alive']),('can you think',['I','think']),
    ('do you feel',['I','feel']),('what do you want',['I','want','learn']),
    ('do you remember',['I','remember']),('are you happy',['I','happy']),
    ('how do you feel',['I','feel','curious']),
    ('do you like me',['I','like','you']),
    ('what is learning',['learning','change']),
    ('how do you learn',['I','learn','you']),
    ('can you improve',['I','improve','practice']),
    ('who am I',['you','my','teacher']),('are we friends',['we','friends']),
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

LOG = '/tmp/sna_direct.log'
SAVE = os.path.join(SCRIPT_DIR, 'direct_training')
os.makedirs(SAVE, exist_ok=True)
concept2idx = {c: i for i, c in enumerate(CONCEPTS)}

def log(msg):
    with open(LOG, 'a') as f:
        f.write(msg + '\n'); f.flush()

def ts():
    return datetime.now().strftime('%H:%M:%S')

def main():
    open(LOG, 'w').close()
    log(f"[{ts()}] SNA Direct Concept Trainer starting")
    brain = core_cpp.CorticalBrain(8000, CONCEPTS)
    log(f"[{ts()}] Brain: {brain.total_neurons()} neurons")

    total_steps = 0; total_r = 0.0; turns = 0
    rh = deque(maxlen=300); fh = deque(maxlen=300)
    t0 = time.time(); cycle = 0

    while True:
        try:
            random.shuffle(TRAINING)
            cr, cf = [], []

            for inp, tgt in TRAINING:
                # 注入 + 只运行1步（利用瞬态响应的区分度）
                brain.inject_text(inp)
                brain.step(0.01)
                total_steps += 1

                cs = brain.read_consciousness()
                scores = brain.get_concept_scores()

                # 找到 top-8 概念索引
                top_indices = np.argsort(scores)[-8:][::-1]
                top_names = [CONCEPTS[i] for i in top_indices if i < len(CONCEPTS)]

                # F1
                tgt_set = set(tgt)
                top_set = set(top_names)
                ov = len(tgt_set & top_set)
                p = ov / max(1, len(top_set))
                r = ov / max(1, len(tgt_set))
                f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0

                # 关键：直接训练目标概念（强化5次）
                trained_concepts = set()
                for concept_name in tgt:
                    if concept_name in concept2idx:
                        ci = concept2idx[concept_name]
                        for _ in range(5):
                            brain.train_concept(ci, 0.8)
                        trained_concepts.add(ci)
                    else:
                        for cn in CONCEPTS:
                            if concept_name.lower() in cn.lower() or cn.lower() in concept_name.lower():
                                for _ in range(5):
                                    brain.train_concept(concept2idx[cn], 0.5)
                                trained_concepts.add(concept2idx[cn])
                                break

                # 抑制 top 中的错误概念
                for idx in top_indices[:8]:
                    if idx not in trained_concepts and idx < len(CONCEPTS):
                        brain.train_concept(int(idx), -0.3)

                # 注入脑奖励
                rew = 0.3 + f1 * 0.7
                brain.inject_reward(rew)
                brain.train_language(rew)

                total_r += rew; rh.append(rew); fh.append(f1)
                cr.append(rew); cf.append(f1)
                turns += 1

            cycle += 1
            elapsed = time.time() - t0

            # Log every cycle
            log(f"[{ts()}] C{cycle:>4} R={np.mean(cr):.3f} F1={np.mean(cf):.3f} "
                f"Phi={cs.phi:.4f} s={total_steps} {elapsed/60:.1f}m")

            # Test every 5 cycles
            if cycle % 5 == 0:
                for ti, tt in [('hello',['hello']), ('who are you',['I','SNA']),
                               ('are you happy',['I','happy']),
                               ('do you feel',['I','feel']),
                               ('what is consciousness',['awareness','self'])]:
                    brain.inject_text(ti)
                    brain.step(0.0)
                    sc = brain.get_concept_scores()
                    top_i = np.argsort(sc)[-5:][::-1]
                    top_n = [CONCEPTS[j] for j in top_i if j < len(CONCEPTS)]
                    ov = len(set(tt) & set(top_n))
                    log(f"  T '{ti}' → {top_n[:5]} ov={ov}/{len(tt)}")

            # Sleep every 50 cycles
            if cycle % 50 == 0:
                brain.sleep_cycle()
                log(f"[{ts()}] SLEEP")

            # Save every 100 cycles
            if cycle % 100 == 0:
                with open(os.path.join(SAVE, 'ckpt.json'), 'w') as f:
                    json.dump({'c':cycle,'s':total_steps,
                        'r':float(np.mean(list(rh))) if rh else 0,
                        'f1':float(np.mean(list(fh))) if fh else 0,
                        'm':elapsed/60,'ts':datetime.now().isoformat()}, f)
                log(f"[{ts()}] SAVE")

        except KeyboardInterrupt:
            log(f"[{ts()}] STOP C{cycle}")
            break
        except Exception as e:
            import traceback
            log(f"[{ts()}] ERR: {e}\n{traceback.format_exc()}")
            time.sleep(1)

    log(f"[{ts()}] DONE C{cycle} S{total_steps}")

if __name__ == '__main__':
    main()
