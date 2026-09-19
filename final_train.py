#!/usr/bin/env v3
"""
SNA Final Trainer v9 — Reservoir Computing with Nonlinear Readout
所有发现的集大成者：
1. 强化文本注入（hash-based + wide cluster）
2. 瞬态响应读取（step=1 最佳区分度）
3. 非线性 readout（2层 MLP）
4. 多时间步特征（step 1,2,3 拼接）
5. brain.train_concept 辅助学习
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


class NonlinearReadout:
    """2层 MLP readout: multi-step thought vector → concept scores"""
    def __input_dim_for(self, tv_dim, n_steps=3):
        return tv_dim * n_steps
    
    def __init__(self, tv_dim, n_concepts, n_steps=3, lr=0.01):
        self.tv_dim = tv_dim
        self.n_concepts = n_concepts
        self.n_steps = n_steps
        self.input_dim = tv_dim * n_steps
        self.hidden_dim = 256
        self.lr = lr
        
        # Xavier init
        self.W1 = np.random.normal(0, np.sqrt(2.0/self.input_dim), 
                    (self.hidden_dim, self.input_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = np.random.normal(0, np.sqrt(2.0/self.hidden_dim),
                    (n_concepts, self.hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(n_concepts, dtype=np.float32)
        
        self.running_mean = np.zeros(self.input_dim, dtype=np.float32)
        self.running_var = np.ones(self.input_dim, dtype=np.float32)
        self.momentum = 0.99
        self._cache = {}
    
    def _norm(self, x):
        self.running_mean = self.momentum * self.running_mean + (1-self.momentum)*x
        self.running_var = self.momentum * self.running_var + (1-self.momentum)*(x-self.running_mean)**2
        return (x - self.running_mean) / (np.sqrt(self.running_var)+1e-8)
    
    def forward(self, x):
        x_n = self._norm(x)
        h = self.W1 @ x_n + self.b1
        h = np.maximum(0, h)  # ReLU
        logits = self.W2 @ h + self.b2
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()+1e-10
        self._cache = {'xn': x_n, 'h': h, 'probs': probs}
        return probs
    
    def train(self, x, target_indices):
        probs = self.forward(x)
        grad = probs.copy()
        for ci in target_indices:
            if ci < self.n_concepts:
                grad[ci] -= 1.0
        
        # Backprop
        gW2 = np.outer(grad, self._cache['h'])
        gb2 = grad
        gh = self.W2.T @ grad
        gh *= (self._cache['h'] > 0)
        gW1 = np.outer(gh, self._cache['xn'])
        gb1 = gh
        
        clip = 1.0
        self.W2 -= self.lr * np.clip(gW2, -clip, clip)
        self.b2 -= self.lr * np.clip(gb2, -clip, clip)
        self.W1 -= self.lr * np.clip(gW1, -clip, clip)
        self.b1 -= self.lr * np.clip(gb1, -clip, clip)
        
        loss = 0.0
        for ci in target_indices:
            if ci < self.n_concepts:
                loss -= np.log(probs[ci]+1e-10)
        return loss / max(1, len(target_indices))
    
    def predict_top_k(self, x, k=8):
        probs = self.forward(x)
        top = np.argsort(probs)[-k:][::-1]
        return [(int(i), float(probs[i])) for i in top]


LOG = '/tmp/sna_final.log'
SAVE = os.path.join(SCRIPT_DIR, 'final_training')
os.makedirs(SAVE, exist_ok=True)

def log(msg):
    with open(LOG, 'a') as f:
        f.write(msg + '\n'); f.flush()

def ts():
    return datetime.now().strftime('%H:%M:%S')

def main():
    open(LOG, 'w').close()
    log(f"[{ts()}] SNA Final Trainer v9")
    
    brain = core_cpp.CorticalBrain(8000, CONCEPTS)
    log(f"[{ts()}] Brain: {brain.total_neurons()} neurons")
    
    # Get thought vector dim
    brain.inject_text('test')
    brain.step(0.0)
    tv_dim = len(brain.read_thought_vector())
    log(f"[{ts()}] TV dim: {tv_dim}")
    
    readout = NonlinearReadout(tv_dim, N_CONCEPTS, n_steps=3, lr=0.015)
    log(f"[{ts()}] Readout: MLP [{tv_dim*3}→256→{N_CONCEPTS}]")
    
    total_steps = 0; turns = 0
    rh = deque(maxlen=500); fh = deque(maxlen=500)
    t0 = time.time(); cycle = 0
    
    while True:
        try:
            random.shuffle(TRAINING)
            cr, cf = [], []
            
            for inp, tgt in TRAINING:
                # Multi-step: collect thought vectors at steps 1,2,3
                brain.inject_text(inp)
                
                tvs = []
                for step_i in range(3):
                    brain.step(0.005)
                    total_steps += 1
                    tv = np.array(brain.read_thought_vector(), dtype=np.float32)
                    tvs.append(tv)
                
                # Concatenate multi-step features
                x = np.concatenate(tvs)
                
                # Target indices
                tgt_idx = []
                for cn in tgt:
                    if cn in concept2idx:
                        tgt_idx.append(concept2idx[cn])
                    else:
                        for c in CONCEPTS:
                            if cn.lower() in c.lower():
                                tgt_idx.append(concept2idx[c]); break
                
                # Train readout
                loss = readout.train(x, tgt_idx)
                
                # Evaluate
                top = readout.predict_top_k(x, k=8)
                top_names = [CONCEPTS[i] for i, _ in top]
                
                ov = len(set(tgt) & set(top_names))
                p = ov / max(1, len(top_names))
                r_ = ov / max(1, len(tgt))
                f1 = 2*p*r_/(p+r_) if (p+r_)>0 else 0.0
                
                # Brain reward
                rew = 0.3 + f1 * 0.7
                brain.inject_reward(rew)
                brain.train_language(rew)
                for ci in tgt_idx:
                    brain.train_concept(ci, 0.3)
                
                rh.append(rew); fh.append(f1)
                cr.append(rew); cf.append(f1)
                turns += 1
            
            cycle += 1
            elapsed = time.time() - t0
            
            log(f"[{ts()}] C{cycle:>4} R={np.mean(cr):.3f} F1={np.mean(cf):.3f} "
                f"L={loss:.3f} {elapsed/60:.1f}m")
            
            if cycle % 5 == 0:
                for ti, tt in [('hello',['hello']), ('who are you',['I','SNA']),
                               ('are you happy',['I','happy']),
                               ('do you feel',['I','feel']),
                               ('what is consciousness',['awareness','self']),
                               ('can you think',['I','think']),
                               ('what is your purpose',['purpose','understand'])]:
                    brain.inject_text(ti)
                    tvs_t = []
                    for _ in range(3):
                        brain.step(0.005)
                        tvs_t.append(np.array(brain.read_thought_vector(), dtype=np.float32))
                    x_t = np.concatenate(tvs_t)
                    top_t = readout.predict_top_k(x_t, k=5)
                    names_t = [CONCEPTS[i] for i, _ in top_t]
                    ov_t = len(set(tt) & set(names_t))
                    log(f"  T '{ti}' → {names_t} ov={ov_t}/{len(tt)}")
            
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
                         W1=readout.W1, b1=readout.b1,
                         W2=readout.W2, b2=readout.b2,
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
