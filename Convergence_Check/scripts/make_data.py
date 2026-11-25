"""Write the synthetic data (services, systems, test sets) used by the convergence check to data/ as JSON.
The simulator regenerates the same systems from these seeds; the files make them inspectable."""
import json
import os

from convcheck.stages import EVAL_SEED, adaptation_set, eval_set
from convcheck.systems import spec_to_dict, stage_a_system, stage_b_system
from convcheck.world import SERVICES

D = "data"
os.makedirs(D, exist_ok=True)


def dump(name, obj):
    with open(os.path.join(D, name), "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", name)


dump("services.json", SERVICES)
dump("stage_A_system.json", spec_to_dict(stage_a_system()))
dump("stage_B_system.json", spec_to_dict(stage_b_system()))
dump("stage_A_test_episodes.json", [dict(seed=sd) for _, sd in eval_set("A")])
dump("stage_B_test_episodes.json", [dict(load_name=name, load=s.load, bursty=s.bursty, seed=sd)
                                    for s, sd, name in eval_set("B")])
dump("stage_C_test_systems.json", [dict(seed=sd, system=spec_to_dict(s)) for s, sd in eval_set("C")])
dump("stage_C_validation_systems.json", [dict(seed=sd, system=spec_to_dict(s)) for s, sd in eval_set("C", 16, EVAL_SEED + 777)])
dump("stage_C_adaptation_pairs.json", [dict(seed=sd, slowed=spec_to_dict(a), normal=spec_to_dict(b))
                                       for a, b, sd in adaptation_set()])
