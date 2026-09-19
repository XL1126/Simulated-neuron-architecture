#!/usr/bin/env python3
"""
SNA Delta Reservoir Trainer v10
关键发现：delta vector（输入后 - 基线）区分度是绝对值的100x！

delta avg cosine = 0.45 vs absolute avg cosine = 0.995

原理：
1. 先采集 baseline（无输入时的 thought vector）
2. 注入文本 → step → 采集 thought vector
3. delta = after - baseline
4. 用 MLP readout 学习 delta → 概念映射
"""
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
    "near","far","I","you","is","not","have","hello","yes","no",
    "what","why","how","learn","happy","sad","want","know","like",
    "see","name","body","neuron","brain","teacher",
    "student","friend","human","alive","consciousness","awareness",
    "understanding","knowledge","emotion","feeling","curiosity",
    "trust","empathy","improve","change","grow","develop","progress",
    "purpose","meaning","different","more","less","before","after",
    "same","start","stop","give","take","make","break",
    "object","food","wall","empty","reward","danger","safe",
    "cat","dog","big","small","fast","slow","hot","cold",
    "up","down","left","right","red","blue","green",
    "light","dark","sound","silence","touch",
    "here","there","now","then","always","never",
    "am","my","your","we","me","too","to","a","are","can",
    "do","with","from","about","SNA","of","for","feedback","result",
    "helps","makes","better","connections","understand","many","things",
    "stored","repeated","trying","becoming","process","through","neural","patterns",
    "in","and","the"
]
concept2idx = {c: i for i, c in enumerate(CONCEPTS)}
N_CONCEPTS = len(CONCEPTS)

TRAINING = [
    ('hello',['hello']),('hi',['hello']),('yes',['yes']),('no',['no']),
    ('good',['good']),('bad',['bad']),('I',['I']),('you',['you']),
    ('self',['self']),('think',['think']),('feel',['feel']),
    ('learn',['learn']),('happy',['happy']),('sad',['sad']),
    ('want',['want']),('know',['know']),('like',['like']),('see',['see']),
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
    ('I am your teacher',['you','my','teacher']),
    ('you are learning',['I','learn','improve']),
    ('think about yourself',['I','self','think']),
    ('what is your brain',['my','brain','neural']),
    ('do you dream',['I','dream','memory']),
    ('wake up',['I','alive','wake']),
    ('I teach you',['you','learn','I','teacher']),
    ('you are my friend',['I','your','friend']),
    ('what is real',['truth','world','self']),
    ('are you happy or sad',['I','happy','not','sad']),
    ('do you like learning',['I','like','learn','happy']),
    ('what is a neuron',['neuron','brain','neural']),
    ('how does your brain work',['brain','neural','connections','patterns']),
]


class MLPReadout:
    def __init__(self, input_dim, n_out, hidden=256, lr=0.01):
        self.lr = lr
        self.W1 = np.random.normal(0, np.sqrt(2.0/input_dim), (hidden, input_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = np.random.normal(0, np.sqrt(2.0/hidden), (n_out, hidden)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)
        self._cache = {}
    
    def forward(self, x):
        h = np.maximum(0, self.W1 @ x + self.b1)
        logits = self.W2 @ h + self.b2
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()+1e-10
        self._cache = {'x': x, 'h': h, 'p': p}
        return p
    
    def train(self, x, targets):
        p = self.forward(x)
        g = p.copy()
        for t in targets:
            if t < len(p): g[t] -= 1.0
        gW2 = np.outer(g, self._cache['h'])
        gh = (self.W2.T @ g) * (self._cache['h'] > 0)
        gW1 = np.outer(gh, self._cache['x'])
        clip = 1.0
        self.W2 -= self.lr * np.clip(gW2, -clip, clip)
        self.b2 -= self.lr * np.clip(g, -clip, clip)
        self.W1 -= self.lr * np.clip(gW1, -clip, clip)
        self.b1 -= self.lr * np.clip(gh, -clip, clip)
        loss = sum(-np.log(p[t]+1e-10) for t in targets if t < len(p))
        return loss / max(1, len(targets))
    
    def top_k(self, x, k=8):
        p = self.forward(x)
        top = np.argsort(p)[-k:][::-1]
        return [(int(i), float(p[i])) for i in top]


LOG = '/tmp/sna_delta.log'
SAVE = os.path.join(SCRIPT_DIR, 'delta_training')
os.makedirs(SAVE, exist_ok=True)

def log(msg):
    with open(LOG, 'a') as f:
        f.write(msg + '\n'); f.flush()

def ts():
    return datetime.now().strftime('%H:%M:%S')

def main():
    open(LOG, 'w').close()
    log(f"[{ts()}] SNA Delta Reservoir Trainer v10")
    
    brain = core_cpp.CorticalBrain(8000, CONCEPTS)
    log(f"[{ts()}] Brain: {brain.total_neurons()} neurons")
    
    # Collect baseline
    brain.step(0.0)
    baseline = np.array(brain.read_thought_vector(), dtype=np.float32)
    tv_dim = len(baseline)
    log(f"[{ts()}] TV dim: {tv_dim}, baseline norm={np.linalg.norm(baseline):.4f}")
    
    readout = MLPReadout(tv_dim, N_CONCEPTS, hidden=256, lr=0.015)
    log(f"[{ts()}] Readout: MLP [{tv_dim}→256→{N_CONCEPTS}]")
    
    total_steps = 0; turns = 0
    fh = deque(maxlen=500)
    t0 = time.time(); cycle = 0
    
    while True:
        try:
            random.shuffle(TRAINING)
            cf = []
            
            for inp, tgt in TRAINING:
                # Inject + 1 step
                brain.inject_text(inp)
                brain.step(0.01)
                total_steps += 1
                
                # DELTA vector
                tv = np.array(brain.read_thought_vector(), dtype=np.float32)
                delta = tv - baseline
                
                # Targets
                tgt_idx = []
                for cn in tgt:
                    if cn in concept2idx: tgt_idx.append(concept2idx[cn])
                    else:
                        for c in CONCEPTS:
                            if cn.lower() in c.lower():
                                tgt_idx.append(concept2idx[c]); break
                
                # Train readout on delta
                loss = readout.train(delta, tgt_idx)
                
                # Evaluate
                top = readout.top_k(delta, k=8)
                top_names = [CONCEPTS[i] for i, _ in top]
                ov = len(set(tgt) & set(top_names))
                p = ov / max(1, len(top_names))
                r_ = ov / max(1, len(tgt))
                f1 = 2*p*r_/(p+r_) if (p+r_)>0 else 0.0
                
                # Brain reward + selective training
                rew = 0.3 + f1 * 0.7
                brain.inject_reward(rew)
                brain.train_language(rew)
                for ci in tgt_idx:
                    brain.train_concept(ci, 0.3)
                
                fh.append(f1); cf.append(f1)
                turns += 1
            
            cycle += 1
            elapsed = time.time() - t0
            
            log(f"[{ts()}] C{cycle:>4} F1={np.mean(cf):.3f} L={loss:.3f} {elapsed/60:.1f}m")
            
            if cycle % 5 == 0:
                for ti, tt in [('hello',['hello']), ('who are you',['I','SNA']),
                               ('are you happy',['I','happy']),
                               ('do you feel',['I','feel']),
                               ('what is consciousness',['awareness','self']),
                               ('can you think',['I','think'])]:
                    brain.inject_text(ti)
                    brain.step(0.01)
                    tv_t = np.array(brain.read_thought_vector(), dtype=np.float32)
                    d_t = tv_t - baseline
                    top_t = readout.top_k(d_t, k=5)
                    names_t = [CONCEPTS[i] for i, _ in top_t]
                    ov_t = len(set(tt) & set(names_t))
                    log(f"  T '{ti}' → {names_t} ov={ov_t}/{len(tt)}")
            
            if cycle % 50 == 0:
                brain.sleep_cycle()
                # Refresh baseline after sleep
                brain.step(0.0)
                baseline = np.array(brain.read_thought_vector(), dtype=np.float32)
                log(f"[{ts()}] SLEEP + baseline refresh")
            
            if cycle % 100 == 0:
                np.savez(os.path.join(SAVE, 'readout.npz'),
                         W1=readout.W1, b1=readout.b1,
                         W2=readout.W2, b2=readout.b2)
                log(f"[{ts()}] SAVE")
        
        except KeyboardInterrupt:
            log(f"[{ts()}] STOP C{cycle}")
            break
        except Exception as e:
            import traceback
            log(f"[{ts()}] ERR: {e}\n{traceback.format_exc()}")
            time.sleep(1)

if __name__ == '__main__':
    main()
