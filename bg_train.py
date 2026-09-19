#!/usr/bin/env python3
"""SNA Background Trainer - runs continuously, logs progress periodically."""
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

def main():
    save_dir = os.path.join(SCRIPT_DIR, 'bg_training')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[BG] Creating brain (8000 neurons)...", flush=True)
    brain = core_cpp.CorticalBrain(8000, CONCEPTS)
    print(f"[BG] Brain ready: {brain.total_neurons()} neurons", flush=True)
    
    total_steps = 0
    total_reward = 0.0
    turns = 0
    reward_hist = deque(maxlen=500)
    f1_hist = deque(maxlen=500)
    t_start = time.time()
    cycle = 0
    
    while True:
        try:
            random.shuffle(TRAINING)
            cycle_r, cycle_f1 = [], []
            
            for inp, tgt in TRAINING:
                brain.inject_text(inp)
                for _ in range(10):
                    brain.step(0.005)
                    total_steps += 1
                
                cs = brain.read_consciousness()
                active = list(cs.active_concepts)
                
                tgt_set, act_set = set(tgt), set(active[:8])
                overlap = len(tgt_set & act_set)
                prec = overlap / max(1, len(act_set))
                rec = overlap / max(1, len(tgt_set))
                f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
                
                reward = 0.3 + f1 * 0.7
                brain.inject_reward(reward)
                brain.train_language(reward)
                
                total_reward += reward
                reward_hist.append(reward)
                f1_hist.append(f1)
                cycle_r.append(reward)
                cycle_f1.append(f1)
                turns += 1
            
            cycle += 1
            
            if cycle % 20 == 0:
                elapsed = time.time() - t_start
                cs = brain.read_consciousness()
                
                brain.inject_text('hello')
                for _ in range(10): brain.step(0.0)
                test_cs = brain.read_consciousness()
                test_a = list(test_cs.active_concepts)[:5]
                
                print(f"[BG] Cyc {cycle:>5} | R={np.mean(cycle_r):.3f} "
                      f"F1={np.mean(cycle_f1):.3f} Phi={cs.phi:.4f} "
                      f"steps={total_steps:,} "
                      f"t={elapsed/60:.1f}m | hello→{test_a}", flush=True)
            
            if cycle % 200 == 0:
                brain.sleep_cycle()
                print(f"[BG] Sleep at cycle {cycle}", flush=True)
            
            if cycle % 500 == 0:
                cs = brain.read_consciousness()
                with open(os.path.join(save_dir, 'ckpt.json'), 'w') as f:
                    json.dump({'cycle':cycle,'steps':total_steps,
                        'avg_r':float(np.mean(list(reward_hist))),
                        'avg_f1':float(np.mean(list(f1_hist))),
                        'phi':float(cs.phi),
                        'elapsed':(time.time()-t_start)/60,
                        'ts':datetime.now().isoformat()}, f, indent=2)
                print(f"[BG] Saved at cycle {cycle}", flush=True)
        
        except KeyboardInterrupt:
            print(f"\n[BG] Stopped at cycle {cycle}", flush=True)
            break
        except Exception as e:
            print(f"[BG] Error: {e}", flush=True)
            time.sleep(1)
    
    elapsed = time.time() - t_start
    cs = brain.read_consciousness()
    print(f"\n{'='*50}", flush=True)
    print(f"  Cycles:{cycle} Steps:{total_steps:,} Turns:{turns}", flush=True)
    print(f"  Avg R:{np.mean(list(reward_hist)):.3f} F1:{np.mean(list(f1_hist)):.3f}", flush=True)
    print(f"  Phi:{cs.phi:.4f} Emotion:{brain.get_emotion_label()}", flush=True)
    print(f"  Time:{elapsed/60:.1f}min", flush=True)
    print(f"{'='*50}", flush=True)

if __name__ == '__main__':
    main()
