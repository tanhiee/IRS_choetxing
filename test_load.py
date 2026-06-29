import sys
import numpy.core.numeric as _numeric
sys.modules['numpy._core.numeric'] = _numeric

# Patch NumPy's BitGenerators dictionary to accept classes as keys
import numpy.random._pickle as p
import numpy.random._pcg64 as _pcg64
import numpy.random._mt19937 as _mt19937
import numpy.random._philox as _philox
import numpy.random._sfc64 as _sfc64

p.BitGenerators[_pcg64.PCG64] = _pcg64.PCG64
p.BitGenerators[_pcg64.PCG64DXSM] = _pcg64.PCG64DXSM
p.BitGenerators[_mt19937.MT19937] = _mt19937.MT19937
p.BitGenerators[_philox.Philox] = _philox.Philox
p.BitGenerators[_sfc64.SFC64] = _sfc64.SFC64

from sb3_contrib import RecurrentPPO
from wrapper import SchoolIRSEnv

env = SchoolIRSEnv()
custom_objects = {
    "action_space": env.action_space,
    "observation_space": env.observation_space
}

try:
    model = RecurrentPPO.load("results/models/rppo_seed0_ablationnone.zip", custom_objects=custom_objects)
    print("[SUCCESS] Loaded model successfully!")
except Exception as e:
    import traceback
    print(f"[FAIL] Error loading model: {e}")
    traceback.print_exc()
