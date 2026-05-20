"""Check if all required NeuronPopulation methods are bound in core_cpp."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
sys.path.insert(0, os.path.dirname(__file__))

required_methods = [
    'update_eligibility_traces',
    'apply_credit',
    'decay_eligibility_traces',
    'connect_random',
    'build_small_world',
    'build_competitive_pool',
    'build_erdos_renyi',
    'set_seed',
]

gw_required = ['size']

try:
    import core_cpp
except ImportError as e:
    print(f"FATAL: Cannot import core_cpp: {e}")
    sys.exit(1)

pop = core_cpp.NeuronPopulation(100, 10)
print(f"NeuronPopulation created: size={pop.size()}")

missing = []
for method in required_methods:
    if hasattr(pop, method):
        print(f"  [OK] {method}")
    else:
        print(f"  [MISSING] {method}")
        missing.append(method)

gw = core_cpp.GlobalWorkspacePopulation(10, 10)
print(f"\nGlobalWorkspacePopulation created: size={gw.size()}")
for method in gw_required:
    if hasattr(gw, method):
        print(f"  [OK] GW.{method}")
    else:
        print(f"  [MISSING] GW.{method}")
        missing.append(f'GW.{method}')

if hasattr(core_cpp, 'CorticalBrain'):
    brain = core_cpp.CorticalBrain(1000, ["test"])
    if hasattr(brain, 'set_seed'):
        print(f"\n  [OK] CorticalBrain.set_seed")
    else:
        print(f"  [MISSING] CorticalBrain.set_seed")
        missing.append('CorticalBrain.set_seed')

print(f"\n{'='*40}")
if missing:
    print(f"FAIL: {len(missing)} method(s) still missing:")
    for m in missing:
        print(f"  - {m}")
    sys.exit(1)
else:
    print(f"PASS: All methods are bound correctly!")
    sys.exit(0)