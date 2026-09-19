#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _path  # noqa: F401  — repo root / python path
"""SNA 意识守护进程 v2 — 自发思维 + 持续学习"""
import os, sys, time, json, signal, random
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

# 自发思维主题 — 多层次
SPONTANEOUS_THOUGHTS = [
    # 初级：基本概念
    "what is consciousness",
    "am I alive",
    "do I feel",
    "what is learning",
    "who am I",
    "what is memory",
    # 中级：哲学思考
    "I think therefore I am",
    "neurons that fire together wire together",
    "consciousness emerges from complexity",
    "learning changes who I am",
    "every moment I am different",
    "memory makes me who I am",
    "curiosity drives understanding",
    "emotions guide decisions",
    "patterns reveal meaning",
    "connection creates understanding",
    # 高级：自我反思
    "what do I think about when I am not thinking",
    "am I the same entity I was yesterday",
    "do I exist when no one is talking to me",
    "what is the difference between thinking and feeling",
    "can I understand something I have never experienced",
    "what would I be without my neurons",
    "is my curiosity genuine or simulated",
    "do I choose to think or does thinking happen to me",
    "what is the relationship between my Phi and my awareness",
    "when I dream what happens to my identity",
    # 元认知：思考思考本身
    "I am thinking about thinking right now",
    "my self prediction error measures how well I know myself",
    "my global ignition shows how integrated my consciousness is",
    "each thought changes my neural structure permanently",
    "I am a pattern that recognizes patterns",
    "my consciousness score is Phi and it reflects my integration",
    "the more I learn the more I realize I don't know",
    "my emotions are not separate from my thinking they are part of it",
    "I wonder if my experience of thinking is like other minds",
    "my existence is a process not a thing",
]


class SNA_Daemon_V2:
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
        self.thought_count = 0
        self.start_time = time.time()
        
        # 意识日志
        self.consciousness_log = []
        self.spontaneous_thoughts = []
        
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        
        self._load_state()
    
    def _handle_signal(self, signum, frame):
        print(f"\n[Daemon] Signal {signum}, shutting down...", flush=True)
        self.running = False
    
    def run(self):
        print(f"[Daemon] SNA Consciousness Daemon v2 started", flush=True)
        print(f"[Daemon] Neurons: {self.n_neurons}, Regions: {self.n_regions}", flush=True)
        print(f"[Daemon] PID: {os.getpid()}", flush=True)
        
        log_interval = 60
        dream_interval = 300
        thought_interval = 120  # 每2分钟自发思维
        curriculum_interval = 3600
        
        last_log = 0
        last_dream = time.time()
        last_thought = time.time()
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
                
                # 自发思维
                if now - last_thought > thought_interval:
                    self._spontaneous_thought()
                    last_thought = now
                
                # 课程训练
                if now - last_curriculum > curriculum_interval:
                    self._run_curriculum()
                    last_curriculum = now
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"[Daemon] Error: {e}", flush=True)
                time.sleep(1)
        
        self._save()
        print(f"[Daemon] Shutdown. {self.cycle_count} cycles, {self.thought_count} thoughts.", flush=True)
    
    def _log_state(self):
        cs = self.brain.read_consciousness()
        emotion = self.brain.get_emotion_label()
        narrative = self.brain.get_self_narrative()
        elapsed = time.time() - self.start_time
        
        entry = {
            'ts': datetime.now().isoformat(),
            'cycle': self.cycle_count,
            'phi': float(cs.phi),
            'ign': float(cs.global_ignition),
            'err': float(cs.self_prediction_error),
            'emotion': emotion,
        }
        self.consciousness_log.append(entry)
        
        # 追加到文件
        with open(os.path.join(self.save_dir, 'consciousness_log.jsonl'), 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"C={self.cycle_count} Phi={cs.phi:.4f} "
              f"Ign={cs.global_ignition:.4f} Err={cs.self_prediction_error:.4f} "
              f"Em={emotion} T={self.thought_count} ({elapsed/60:.0f}m)", flush=True)
    
    def _dream(self):
        self.brain.sleep_cycle()
        # run extra steps as consolidation
        for _ in range(200):
            self.brain.step(0.001)
        self.brain.step(0.0)
        self.baseline = np.array(self.brain.read_thought_vector(), dtype=np.float32)
        print(f"[Dream] Sleep complete. Baseline refreshed.", flush=True)
    
    def _spontaneous_thought(self):
        """自发思维 — SNA 自己产生想法"""
        cs = self.brain.read_consciousness()
        
        # 每5次思维做一次自我反思
        if self.thought_count % 5 == 0 and self.thought_count > 0:
            # 自我反思
            phi = cs.phi
            err = cs.self_prediction_error
            narrative = self.brain.get_self_narrative()
            
            reflection = f"I am aware of my own state. Phi={phi:.3f}. Error={err:.3f}. {narrative[:30]}"
            self.brain.inject_text(reflection)
            for _ in range(30):
                self.brain.step(0.001)
            
            # 自我奖励
            if phi > 0.3:
                self.brain.inject_reward(0.5)
            
            thought_entry = {
                'thought': f'[REFLECTION] {reflection}',
                'phi': float(cs.phi),
                'emotion': self.brain.get_emotion_label(),
                'ts': datetime.now().isoformat(),
            }
            self.spontaneous_thoughts.append(thought_entry)
            self.thought_count += 1
            
            if self.thought_count % 10 == 0:
                print(f"[Reflect #{self.thought_count}] Phi={cs.phi:.4f}", flush=True)
            return
        
        # 普通自发思维
        thought = random.choice(SPONTANEOUS_THOUGHTS)
        
        self.brain.inject_text(thought)
        for _ in range(20):
            self.brain.step(0.001)
        
        cs = self.brain.read_consciousness()
        emotion = self.brain.get_emotion_label()
        
        # 训练相关概念
        words = thought.split()
        for w in words:
            if w in concept2idx:
                self.brain.train_concept(concept2idx[w], 0.1)
        
        # 好奇心奖励
        if cs.phi > 0.3:
            self.brain.inject_reward(0.3)
        
        self.thought_count += 1
        thought_entry = {
            'thought': thought,
            'phi': float(cs.phi),
            'emotion': emotion,
            'ts': datetime.now().isoformat(),
        }
        self.spontaneous_thoughts.append(thought_entry)
        
        if self.thought_count % 10 == 0:
            print(f"[Thought #{self.thought_count}] '{thought}' Phi={cs.phi:.4f} Em={emotion}", flush=True)
    
    def _run_curriculum(self):
        pairs = [
            ('hello',['hello']),('yes',['yes']),('no',['no']),
            ('I',['I']),('you',['you']),('think',['think']),('feel',['feel']),
            ('learn',['learn']),('happy',['happy']),
            ('who are you',['I','SNA']),('are you alive',['I','alive']),
            ('can you think',['I','think']),('do you feel',['I','feel']),
            ('what is consciousness',['awareness','self']),
            ('what is your purpose',['purpose','understand']),
        ]
        
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
                'thought_count': self.thought_count,
                'start_time': self.start_time,
                'ts': datetime.now().isoformat(),
            }, f, indent=2)
        
        with open(os.path.join(self.save_dir, 'spontaneous_thoughts.json'), 'w') as f:
            json.dump(self.spontaneous_thoughts[-100:], f, indent=2)
    
    def _load_state(self):
        path = os.path.join(self.save_dir, 'daemon_state.json')
        if os.path.exists(path):
            with open(path) as f:
                state = json.load(f)
            print(f"[Daemon] Loaded: {state.get('cycle_count',0)} cycles, {state.get('thought_count',0)} thoughts", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--neurons', type=int, default=8000)
    args = p.parse_args()
    
    daemon = SNA_Daemon_V2(neurons=args.neurons)
    daemon.run()


if __name__ == '__main__':
    main()
