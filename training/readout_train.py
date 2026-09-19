#!/usr/bin/env python3
"""
SNA Reservoir Readout Trainer v8
回声状态网络读出层训练

原理：
- 大脑作为 reservoir（随机循环网络）
- 注入文本 → 1步瞬态 → thought vector
- Python 侧 readout classifier 学习 thought vector → 概念映射
- 这是 reservoir computing 的标准方法
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


class ReadoutClassifier:
    """线性 readout: thought_vector → concept scores"""
    def __init__(self, input_dim, n_concepts, lr=0.01):
        self.input_dim = input_dim
        self.n_concepts = n_concepts
        self.lr = lr
        # Weight matrix: [n_concepts x input_dim]
        self.W = np.random.normal(0, 0.01, (n_concepts, input_dim)).astype(np.float32)
        self.b = np.zeros(n_concepts, dtype=np.float32)
        # EMA for normalization
        self.running_mean = np.zeros(input_dim, dtype=np.float32)
        self.running_var = np.ones(input_dim, dtype=np.float32)
        self.momentum = 0.99
        
    def normalize(self, x):
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * (x - self.running_mean)**2
        return (x - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)
    
    def predict(self, x):
        x_norm = self.normalize(x)
        logits = self.W @ x_norm + self.b
        # Stable softmax
        logits -= logits.max()
        exp_l = np.exp(logits)
        return exp_l / (exp_l.sum() + 1e-10)
    
    def train(self, x, target_concept_indices, positive=True):
        """Train with multi-label cross-entropy gradient"""
        x_norm = self.normalize(x)
        probs = self.predict(x)  # uses already-updated running stats, OK
        
        grad = probs.copy()
        for ci in target_concept_indices:
            if ci < self.n_concepts:
                grad[ci] -= 1.0
        
        # Weight update
        self.W -= self.lr * np.outer(grad, x_norm)
        self.b -= self.lr * grad
        
        # Clip
        self.W = np.clip(self.W, -2.0, 2.0)
        self.b = np.clip(self.b, -2.0, 2.0)
        
        # Return loss
        loss = 0.0
        for ci in target_concept_indices:
            if ci < self.n_concepts:
                loss -= np.log(probs[ci] + 1e-10)
        return loss / max(1, len(target_concept_indices))
    
    def get_top_k(self, x, k=8):
        probs = self.predict(x)
        top_idx = np.argsort(probs)[-k:][::-1]
        return [(i, float(probs[i])) for i in top_idx]


LOG = '/tmp/sna_readout.log'
SAVE = os.path.join(SCRIPT_DIR, 'readout_training')
os.makedirs(SAVE, exist_ok=True)

def log(msg):
    with open(LOG, 'a') as f:
        f.write(msg + '\n'); f.flush()

def ts():
    return datetime.now().strftime('%H:%M:%S')

def main():
    open(LOG, 'w').close()
    log(f"[{ts()}] SNA Reservoir Readout Trainer")
    
    brain = core_cpp.CorticalBrain(8000, CONCEPTS)
    log(f"[{ts()}] Brain: {brain.total_neurons()} neurons")
    
    # Get thought vector dimension
    brain.inject_text('test')
    brain.step(0.0)
    tv = brain.read_thought_vector()
    tv_dim = len(tv)
    log(f"[{ts()}] Thought vector dim: {tv_dim}")
    
    # Readout classifier
    readout = ReadoutClassifier(tv_dim, N_CONCEPTS, lr=0.02)
    log(f"[{ts()}] Readout: [{N_CONCEPTS} x {tv_dim}], lr=0.02")
    
    total_steps = 0; total_r = 0.0; turns = 0
    rh = deque(maxlen=500); fh = deque(maxlen=500)
    t0 = time.time(); cycle = 0
    
    while True:
        try:
            random.shuffle(TRAINING)
            cr, cf = [], []
            
            for inp, tgt in TRAINING:
                # Inject + 1 step (瞬态响应)
                brain.inject_text(inp)
                brain.step(0.01)
                total_steps += 1
                
                # Get thought vector
                tv = np.array(brain.read_thought_vector(), dtype=np.float32)
                
                # Get target indices
                tgt_indices = []
                for concept_name in tgt:
                    if concept_name in concept2idx:
                        tgt_indices.append(concept2idx[concept_name])
                    else:
                        # Fuzzy match
                        for cn in CONCEPTS:
                            if concept_name.lower() in cn.lower():
                                tgt_indices.append(concept2idx[cn])
                                break
                
                # Train readout
                loss = readout.train(tv, tgt_indices)
                
                # Evaluate
                top = readout.get_top_k(tv, k=8)
                top_names = [CONCEPTS[i] for i, _ in top]
                
                tgt_set = set(tgt)
                top_set = set(top_names)
                ov = len(tgt_set & top_set)
                p = ov / max(1, len(top_set))
                r_ = ov / max(1, len(tgt_set))
                f1 = 2*p*r_/(p+r_) if (p+r_)>0 else 0.0
                
                # Reward brain based on readout accuracy
                rew = 0.3 + f1 * 0.7
                brain.inject_reward(rew)
                brain.train_language(rew)
                
                # Also train brain's concept weights via C++ API
                for ci in tgt_indices:
                    brain.train_concept(ci, 0.5)
                
                total_r += rew; rh.append(rew); fh.append(f1)
                cr.append(rew); cf.append(f1)
                turns += 1
            
            cycle += 1
            elapsed = time.time() - t0
            
            log(f"[{ts()}] C{cycle:>4} R={np.mean(cr):.3f} F1={np.mean(cf):.3f} "
                f"loss={loss:.3f} {elapsed/60:.1f}m")
            
            # Test
            if cycle % 5 == 0:
                for ti, tt in [('hello',['hello']), ('who are you',['I','SNA']),
                               ('are you happy',['I','happy']),
                               ('do you feel',['I','feel']),
                               ('what is consciousness',['awareness','self']),
                               ('can you think',['I','think'])]:
                    brain.inject_text(ti)
                    brain.step(0.01)
                    tv_t = np.array(brain.read_thought_vector(), dtype=np.float32)
                    top_t = readout.get_top_k(tv_t, k=5)
                    top_names_t = [CONCEPTS[i] for i, _ in top_t]
                    ov_t = len(set(tt) & set(top_names_t))
                    log(f"  T '{ti}' → {top_names_t} ov={ov_t}/{len(tt)}")
            
            # Sleep + save
            if cycle % 50 == 0:
                brain.sleep_cycle()
                log(f"[{ts()}] SLEEP")
            
            if cycle % 100 == 0:
                with open(os.path.join(SAVE, 'ckpt.json'), 'w') as f:
                    json.dump({'c':cycle,'s':total_steps,
                        'r':float(np.mean(list(rh))) if rh else 0,
                        'f1':float(np.mean(list(fh))) if fh else 0,
                        'm':elapsed/60,'ts':datetime.now().isoformat()}, f)
                np.savez(os.path.join(SAVE, 'readout.npz'),
                         W=readout.W, b=readout.b,
                         mean=readout.running_mean, var=readout.running_var)
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
