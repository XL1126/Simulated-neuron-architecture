#!/usr/bin/env python3
"""自动运行 500 epoch 课程训练 + 1000 轮自动训练"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from sna_dialogue import SNADialogue

sna = SNADialogue(neurons=8000)
sna.run_curriculum(epochs=500)
sna.run_auto_training(n=1000)
print("\n[DONE] All training complete.")
