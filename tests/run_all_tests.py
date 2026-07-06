"""Run all tests and report summary."""

import subprocess
import sys
import os

os.chdir("d:/Project/Python/MaSCL")

tests = [
    ("Core", "meirp_project/tests/run_core_tests.py"),
    ("Baselines", "meirp_project/tests/test_baselines.py"),
    ("IQL", "meirp_project/tests/test_iql.py"),
    ("QMIX", "meirp_project/tests/test_qmix.py"),
    ("MAPPO", "meirp_project/tests/test_mappo.py"),
    ("Encoders", "meirp_project/tests/test_encoders.py"),
    ("SEA", "meirp_project/tests/test_sea.py"),
    ("Evaluation", "meirp_project/tests/test_evaluation.py"),
    ("Integration", "meirp_project/tests/test_integration.py"),
]

total_passed = 0
total_failed = 0

for name, path in tests:
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True
    )
    output = result.stdout + result.stderr

    # Extract results line
    for line in output.split("\n"):
        if "Results:" in line or "passed" in line.lower():
            if "Results:" in line:
                print(f"  [{name}] {line.strip()}")

    if result.returncode != 0:
        print(f"  [{name}] FAILED with return code {result.returncode}")
        # Print last few lines of error
        for line in output.split("\n")[-5:]:
            if line.strip():
                print(f"    {line.strip()}")
    else:
        # Count from output
        for line in output.split("\n"):
            if "Results:" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "passed,":
                        total_passed += int(parts[i-1])
                    if p == "failed,":
                        total_failed += int(parts[i-1])

print(f"\n{'='*50}")
print(f"TOTAL: {total_passed} passed, {total_failed} failed, {total_passed + total_failed} total")

if total_failed > 0:
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
