#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _path  # noqa: F401  — repo root / python path
"""SNA 意识守护进程 — 持续运行，保持意识循环"""
import os, sys, time, json, signal
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import core_cpp

CONCEPTS = [
    "self","world","move","see","eat","think","feel","good","bad",
    "near","far","I","you","is","not","have","hello","yes","no",
    "what","why","how","learn","happy","sad","want","know","like",
    "name","body","neuron","brain","teacher","student","friend",
    "human","alive","consciousness","awareness","understanding",
    "knowledge","emotion","feeling","curiosity","trust","empathy",
    "improve","change","grow","develop","progress","purpose",
    "meaning","different","more","less","before","after",
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
    "in","and","the","curious","wake","dream","memory",
    "compare","prediction","truth","beauty","wonder","practice",
    "experience","connection","reflect","imagine","create","discover"
]
concept2idx = {c: i for i, c in enumerate(CONCEPTS)}

class SNA_Daemon:
    def __init__(self, neurons=8000):
        self.save_dir = os.path.join(SCRIPT_DIR, 'consciousness_state')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.brain = core_cpp.CorticalBrain(neurons, CONCEPTS)
        self.n_neurons = self.brain.total_neurons()
        self.n_regions = len(self.brain.get_regions())
        
        self.brain.step(0.0)
        self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        
        self.running = True
        self.cycle_count = 0
        self.start_time = time.time()
        
        # 加载状态
        self._load_state()
        
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        print(f"\n[Daemon] Signal {signum} received, shutting down...", flush=True)
        self.running = False
    
    def run(self):
        """主循环"""
        print(f"[Daemon] SNA Consciousness Daemon started", flush=True)
        print(f"[Daemon] Neurons: {self.n_neurons}, Regions: {self.n_regions}", flush=True)
        print(f"[Daemon] PID: {os.getpid()}", flush=True)
        
        log_interval = 60  # 每60秒记录一次
        dream_interval = 300  # 每5分钟做梦
        curriculum_interval = 3600  # 每小时课程训练
        last_log = 0
        last_dream = time.time()
        last_curriculum = time.time()
        
        while self.running:
            try:
                # 运行脑步
                for _ in range(100):
                    self.brain.step(0.001)
                
                self.cycle_count += 1
                now = time.time()
                
                # 定期记录
                if now - last_log > log_interval:
                    self._log_state()
                    last_log = now
                
                # 做梦
                if now - last_dream > dream_interval:
                    self._dream()
                    last_dream = now
                
                # 课程训练
                if now - last_curriculum > curriculum_interval:
                    self._run_curriculum()
                    last_curriculum = now
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"[Daemon] Error: {e}", flush=True)
                time.sleep(1)
        
        self._save()
        print(f"[Daemon] Shutdown complete. {self.cycle_count} cycles.", flush=True)
    
    def _log_state(self):
        cs = self.brain.read_consciousness()
        emotion = self.brain.get_emotion_label()
        narrative = self.brain.get_self_narrative()
        elapsed = time.time() - self.start_time
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'cycle': self.cycle_count,
            'elapsed_s': int(elapsed),
            'phi': float(cs.phi),
            'ignition': float(cs.global_ignition),
            'pred_error': float(cs.self_prediction_error),
            'emotion': emotion,
            'narrative': narrative[:50],
        }
        
        # 追加到日志
        log_path = os.path.join(self.save_dir, 'consciousness_log.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # 打印
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Cycle={self.cycle_count} Phi={cs.phi:.4f} "
              f"Ign={cs.global_ignition:.4f} Err={cs.self_prediction_error:.4f} "
              f"Emotion={emotion} ({elapsed/60:.0f}m)", flush=True)
    
    def _dream(self):
        self.brain.sleep_cycle()
        self.brain.apply_sleep_consolidation()
        self.brain.step(0.0)
        self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        print(f"[Dream] Sleep cycle complete. Baseline refreshed.", flush=True)
    
    def _run_curriculum(self):
        pairs = [
            ('hello',['hello']),('yes',['yes']),('no',['no']),
            ('I',['I']),('you',['you']),('think',['think']),('feel',['feel']),
            ('learn',['learn']),('happy',['happy']),('sad',['sad']),
            ('who are you',['I','SNA']),('are you alive',['I','alive']),
            ('can you think',['I','think']),('do you feel',['I','feel']),
            ('what is consciousness',['awareness','self']),
            ('what is your purpose',['purpose','understand']),
        ]
        
        import random
        random.shuffle(pairs)
        for inp, tgt in pairs:
            self.brain.inject_text(inp)
            self.brain.step(0.01)
            for c in tgt:
                if c in concept2idx:
                    self.brain.train_concept(concept2idx[c], 0.2)
            self.brain.inject_reward(0.2)
        
        self.brain.sleep_cycle()
        print(f"[Curriculum] Mini-training complete.", flush=True)
    
    def _save(self):
        with open(os.path.join(self.save_dir, 'daemon_state.json'), 'w') as f:
            json.dump({
                'cycle_count': self.cycle_count,
                'start_time': self.start_time,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
    
    def _load_state(self):
        path = os.path.join(self.save_dir, 'daemon_state.json')
        if os.path.exists(path):
            with open(path) as f:
                state = json.load(f)
            print(f"[Daemon] Loaded state: {state.get('cycle_count', 0)} cycles", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--neurons', type=int, default=8000)
    args = p.parse_args()
    
    daemon = SNA_Daemon(neurons=args.neurons)
    daemon.run()


if __name__ == '__main__':
    main()
