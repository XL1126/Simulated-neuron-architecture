#!/usr/bin/env python3
"""
SNA Frozen Reservoir Trainer v11
完全冻结 reservoir，只训练 readout

关键改动：
1. 不调用 brain.inject_reward / train_language / train_concept
2. baseline 固定不变（不随 sleep 刷新）
3. 学习率更低（0.005），梯度裁剪更严格（0.5）
4. Adam-like momentum for stability
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
N = len(CONCEPTS)

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


class AdamReadout:
    """MLP with Adam optimizer for stable training"""
    def __init__(self, d_in, d_out, d_h=256, lr=0.005):
        self.lr = lr
        scale1 = np.sqrt(2.0/d_in)
        scale2 = np.sqrt(2.0/d_h)
        self.W1 = np.random.normal(0, scale1, (d_h, d_in)).astype(np.float32)
        self.b1 = np.zeros(d_h, dtype=np.float32)
        self.W2 = np.random.normal(0, scale2, (d_out, d_h)).astype(np.float32)
        self.b2 = np.zeros(d_out, dtype=np.float32)
        # Adam moments
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.t = 0
        self._cache = {}
    
    def forward(self, x):
        h = np.maximum(0, self.W1 @ x + self.b1)
        logits = self.W2 @ h + self.b2
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()+1e-10
        self._cache = {'x': x, 'h': h, 'p': p}
        return p
    
    def _adam_update(self, param_name, grad):
        m = getattr(self, f'm{param_name}')
        v = getattr(self, f'v{param_name}')
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = beta1 * m + (1-beta1) * grad
        v = beta2 * v + (1-beta2) * grad**2
        m_hat = m / (1 - beta1**self.t)
        v_hat = v / (1 - beta2**self.t)
        update = self.lr * m_hat / (np.sqrt(v_hat) + eps)
        setattr(self, f'm{param_name}', m)
        setattr(self, f'v{param_name}', v)
        return update
    
    def train(self, x, targets):
        self.t += 1
        p = self.forward(x)
        g = p.copy()
        for t in targets:
            if t < len(p): g[t] -= 1.0
        
        gW2 = np.outer(g, self._cache['h'])
        gh = (self.W2.T @ g) * (self._cache['h'] > 0)
        gW1 = np.outer(gh, self._cache['x'])
        
        clip = 0.5
        self.W2 -= np.clip(self._adam_update('W2', np.clip(gW2,-clip,clip)), -clip, clip)
        self.b2 -= np.clip(self._adam_update('b2', np.clip(g,-clip,clip)), -clip, clip)
        self.W1 -= np.clip(self._adam_update('W1', np.clip(gW1,-clip,clip)), -clip, clip)
        self.b1 -= np.clip(self._adam_update('b1', np.clip(gh,-clip,clip)), -clip, clip)
        
        loss = sum(-np.log(p[t]+1e-10) for t in targets if t < len(p))
        return loss / max(1, len(targets))
    
    def top_k(self, x, k=8):
        p = self.forward(x)
        top = np.argsort(p)[-k:][::-1]
        return [(int(i), float(p[i])) for i in top]


LOG = '/tmp/sna_frozen.log'
SAVE = os.path.join(SCRIPT_DIR, 'frozen_training')
os.makedirs(SAVE, exist_ok=True)

def log(msg):
    with open(LOG, 'a') as f:
        f.write(msg + '\n'); f.flush()

def ts():
    return datetime.now().strftime('%H:%M:%S')

def main():
    open(LOG, 'w').close()
    log(f"[{ts()}] SNA Frozen Reservoir Trainer v11")
    
    brain = core_cpp.CorticalBrain(8000, CONCEPTS)
    log(f"[{ts()}] Brain: {brain.total_neurons()} neurons")
    
    # Collect FIXED baseline (never changes)
    brain.step(0.0)
    baseline = np.array(brain.read_thought_vector(), dtype=np.float32)
    d = len(baseline)
    log(f"[{ts()}] TV dim: {d}")
    
    readout = AdamReadout(d, N, d_h=256, lr=0.005)
    log(f"[{ts()}] Readout: Adam-MLP [{d}→256→{N}] lr=0.005")
    
    # Pre-collect all delta vectors (freeze reservoir snapshots)
    log(f"[{ts()}] Pre-collecting delta vectors...")
    snapshots = []
    for inp, tgt in TRAINING:
        brain.inject_text(inp)
        brain.step(0.01)
        tv = np.array(brain.read_thought_vector(), dtype=np.float32)
        delta = tv - baseline
        
        tgt_idx = []
        for cn in tgt:
            if cn in concept2idx: tgt_idx.append(concept2idx[cn])
            else:
                for c in CONCEPTS:
                    if cn.lower() in c.lower():
                        tgt_idx.append(concept2idx[c]); break
        
        snapshots.append((delta, tgt_idx, inp, tgt))
    log(f"[{ts()}] Collected {len(snapshots)} snapshots")
    
    # Train readout on frozen snapshots
    t0 = time.time()
    for epoch in range(2000):
        random.shuffle(snapshots)
        losses = []
        f1s = []
        
        for delta, tgt_idx, inp, tgt in snapshots:
            loss = readout.train(delta, tgt_idx)
            losses.append(loss)
            
            top = readout.top_k(delta, k=8)
            top_names = [CONCEPTS[i] for i, _ in top]
            ov = len(set(tgt) & set(top_names))
            p_ = ov / max(1, len(top_names))
            r_ = ov / max(1, len(tgt))
            f1 = 2*p_*r_/(p_+r_) if (p_+r_)>0 else 0.0
            f1s.append(f1)
        
        if epoch % 20 == 0 or epoch == 1999:
            avg_loss = np.mean(losses)
            avg_f1 = np.mean(f1s)
            
            # Test on specific samples
            test_lines = []
            for ti, tt in [('hello',['hello']), ('who are you',['I','SNA']),
                           ('are you happy',['I','happy']),
                           ('do you feel',['I','feel']),
                           ('what is consciousness',['awareness','self']),
                           ('can you think',['I','think'])]:
                # Find this snapshot
                for delta_s, tgt_s, inp_s, tgt_n in snapshots:
                    if inp_s == ti:
                        top_t = readout.top_k(delta_s, k=5)
                        names_t = [CONCEPTS[i] for i, _ in top_t]
                        ov_t = len(set(tt) & set(names_t))
                        test_lines.append(f"  '{ti}' → {names_t} ov={ov_t}/{len(tt)}")
                        break
            
            log(f"[{ts()}] E{epoch:>5} F1={avg_f1:.3f} L={avg_loss:.3f} "
                f"{(time.time()-t0)/60:.1f}m")
            for tl in test_lines:
                log(tl)
        
        if epoch % 200 == 0 and epoch > 0:
            np.savez(os.path.join(SAVE, 'readout.npz'),
                     W1=readout.W1, b1=readout.b1,
                     W2=readout.W2, b2=readout.b2)
            log(f"[{ts()}] SAVE")
    
    np.savez(os.path.join(SAVE, 'readout_final.npz'),
             W1=readout.W1, b1=readout.b1,
             W2=readout.W2, b2=readout.b2)
    log(f"[{ts()}] DONE. Final F1={np.mean(f1s):.3f}")

if __name__ == '__main__':
    main()
