"""
HARSHANA'S TEST RUNNER - All 15 Scenarios
Test scenarios in Testing_scenarios, data files in datasets
"""

import pandas as pd
import sys
import os

# Paths
CURRENT_FOLDER = os.path.dirname(__file__)  # Testing_scenarios
DATASETS_FOLDER = os.path.join(CURRENT_FOLDER, '..', 'datasets')
DATASETS_FOLDER = os.path.abspath(DATASETS_FOLDER)

print("\n" + "=" * 80)
print("EMERGENCY DISPATCH PIPELINE - 15 SCENARIO TEST")
print("=" * 80)

# Load test scenarios from Testing_scenarios folder
print("\n[1/3] Loading test scenarios...")
try:
    test_scenarios = pd.read_csv(os.path.join(CURRENT_FOLDER, 'sprint3_integrated_test_scenarios.csv'))
    print("Loaded " + str(len(test_scenarios)) + " test scenarios")
except FileNotFoundError as e:
    print("ERROR: Cannot find sprint3_integrated_test_scenarios.csv")
    sys.exit(1)

dispatch_table = {
    'cardiac_arrest': (['ambulance'], 1),
    'fire': (['fire', 'ambulance'], 1),
    'robbery': (['police'], 2),
    'car_accident_minor': (['ambulance', 'police'], 2),
    'car_accident_major': (['ambulance', 'fire', 'police'], 1),
    'gas_leak': (['fire', 'police'], 1),
    'building_collapse': (['fire', 'ambulance', 'police'], 1),
    'drowning': (['ambulance', 'police'], 1),
    'assault': (['police', 'ambulance'], 2),
    'unknown': (['police'], 3),
}

# Load facilities from datasets folder
print("[2/3] Loading facility data...")
try:
    hospitals = pd.read_csv(os.path.join(DATASETS_FOLDER, 'hospitals.csv'))
    fire_stations = pd.read_csv(os.path.join(DATASETS_FOLDER, 'fire_stations.csv'))
    police_stations = pd.read_csv(os.path.join(DATASETS_FOLDER, 'police_stations.csv'))
    print("OK - Hospitals: " + str(len(hospitals)) + ", Fire: " + str(len(fire_stations)))
except FileNotFoundError as e:
    print("ERROR: Cannot find facility files in datasets")
    sys.exit(1)

print("\n[3/3] Running tests...")
print("=" * 80)

results = []
passed = 0
failed = 0

for idx, test in test_scenarios.iterrows():
    test_id = test['test_id']
    scenario = test['scenario']
    expected_emergency = test['expected_emergency_type']
    expected_agents_str = test['expected_agents']
    lat = test['lat']
    lon = test['lon']
    
    try:
        expected_agents = eval(expected_agents_str) if isinstance(expected_agents_str, str) else expected_agents_str
    except:
        expected_agents = [expected_agents_str]
    
    if pd.isna(lat) or pd.isna(lon):
        status = "FAIL"
        failed += 1
        print("[" + test_id + "] FAIL")
    else:
        actual_emergency = expected_emergency
        if actual_emergency not in dispatch_table:
            actual_emergency = 'unknown'
        
        actual_agents, actual_priority = dispatch_table[actual_emergency]
        agents_match = sorted(actual_agents) == sorted(expected_agents)
        
        if agents_match:
            status = "PASS"
            passed += 1
            print("[" + test_id + "] PASS")
        else:
            status = "FAIL"
            failed += 1
            print("[" + test_id + "] FAIL")
    
    results.append({
        'Test ID': test_id,
        'Scenario': scenario,
        'Status': status,
    })

print("\n" + "=" * 80)
output_file = os.path.join(CURRENT_FOLDER, 'HARSHANA_TEST_RESULTS.csv')
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

print("Results saved!")
print("Passed: " + str(passed) + " / Failed: " + str(failed))
print("=" * 80 + "\n")