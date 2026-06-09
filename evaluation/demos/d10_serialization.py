"""D10: distributed serialization round-trip + standalone codegen."""
from asymsafety.beta.einstein_hilbert import build_eh_beta_system
from asymsafety.compute.distributed.serialization import (
    serialize_to_json, deserialize_from_json, generate_standalone_code)
from asymsafety.analysis.fixed_points import FixedPointFinder

system = build_eh_beta_system(d=4)
fp0 = FixedPointFinder(system).find_fixed_point({"g": 0.7, "lambda": 0.14})

js = serialize_to_json(system)
print(f"serialized JSON: {len(js)} bytes")
restored = deserialize_from_json(js)
fp1 = FixedPointFinder(restored).find_fixed_point({"g": 0.7, "lambda": 0.14})

dg = abs(fp0.location["g"] - fp1.location["g"])
dl = abs(fp0.location["lambda"] - fp1.location["lambda"])
print(f"FP drift after round-trip: dg={dg:.3e}, dl={dl:.3e}")
print(f"VERDICT round-trip: {'PASS' if dg < 1e-12 and dl < 1e-12 else 'FAIL'}")

code = generate_standalone_code(system)
print(f"standalone code: {len(code)} chars; contains def: {'def ' in code}")
