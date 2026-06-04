#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

def parse_log_file(filepath):
    """Parse log file and extract train/test/pr data"""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse train loss
            if line.startswith('Train | Epoch:'):
                match = re.match(r'Train \| Epoch: (\d+) \| Loss: ([\d.]+)', line)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    if epoch not in data:
                        data[epoch] = {}
                    data[epoch]['train_loss'] = loss

            # Parse test MAP
            elif line.startswith('Test | Epoch'):
                match = re.match(r'Test \| Epoch (\d+) \| MAP: ([\d.]+) \| Best MAP: ([\d.]+)', line)
                if match:
                    epoch = int(match.group(1))
                    map_val = float(match.group(2))
                    best_map = float(match.group(3))
                    if epoch not in data:
                        data[epoch] = {}
                    data[epoch]['test_map'] = map_val
                    data[epoch]['best_map'] = best_map

            # Parse PR values
            elif line.startswith('PR | Epoch'):
                match = re.match(r'PR \| Epoch (\d+) \| (.+)', line)
                if match:
                    epoch = int(match.group(1))
                    values_str = match.group(2)
                    values = [float(v) for v in values_str.split()]
                    if epoch not in data:
                        data[epoch] = {}
                    data[epoch]['pr_values'] = values

    return data

def compare_values(log_val, json_val, name, tolerance=0.001):
    """Compare two float values with tolerance"""
    if json_val is None:
        return f"❌ {name}: JSON missing (log has {log_val})"

    if isinstance(json_val, list):
        if len(json_val) != len(log_val):
            return f"❌ {name}: Length mismatch - log: {len(log_val)}, json: {len(json_val)}"

        mismatches = []
        for i, (l, j) in enumerate(zip(log_val, json_val)):
            if abs(l - j) > tolerance:
                mismatches.append(f"[{i}] log={l:.5f}, json={j:.5f}")

        if mismatches:
            return f"❌ {name}: {', '.join(mismatches[:3])}" + (f"... +{len(mismatches)-3} more" if len(mismatches) > 3 else "")
        return f"✓ {name}"
    else:
        if abs(log_val - json_val) > tolerance:
            return f"❌ {name}: log={log_val:.4f}, json={json_val:.4f}"
        return f"✓ {name}"

def verify_scenario(scenario_path):
    """Verify all log files in a scenario"""
    print(f"\n{'='*60}")
    print(f"Checking scenario: {scenario_path.name}")
    print(f"{'='*60}")

    # Load JSON summary
    json_file = scenario_path / "summary.json"
    if not json_file.exists():
        print(f"❌ JSON file not found: {json_file}")
        return

    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Check each log file
    log_files = sorted(scenario_path.glob("*.txt"))

    errors_found = False
    for log_file in log_files:
        model_name = log_file.stem
        print(f"\n📄 {model_name}")

        if model_name not in json_data:
            print(f"  ❌ Not found in JSON")
            errors_found = True
            continue

        log_data = parse_log_file(log_file)
        json_model_data = json_data[model_name]

        # Extract epochs from JSON
        json_epochs = {}
        for entry in json_model_data:
            epoch = entry.get('epoch')
            if epoch not in json_epochs:
                json_epochs[epoch] = {}

            if entry['type'] == 'train':
                json_epochs[epoch]['train_loss'] = entry.get('loss')
            elif entry['type'] == 'test':
                json_epochs[epoch]['test_map'] = entry.get('map')
                json_epochs[epoch]['best_map'] = entry.get('best_map')
            elif entry['type'] == 'pr':
                json_epochs[epoch]['pr_values'] = entry.get('values')

        # Compare data
        all_epochs = set(log_data.keys()) | set(json_epochs.keys())
        issues_count = 0

        for epoch in sorted(all_epochs):
            if epoch not in log_data:
                print(f"  ❌ Epoch {epoch}: Missing in log file")
                issues_count += 1
                errors_found = True
                continue

            if epoch not in json_epochs:
                print(f"  ❌ Epoch {epoch}: Missing in JSON")
                issues_count += 1
                errors_found = True
                continue

            log_epoch = log_data[epoch]
            json_epoch = json_epochs[epoch]

            # Check train loss
            if 'train_loss' in log_epoch:
                result = compare_values(
                    log_epoch['train_loss'],
                    json_epoch.get('train_loss'),
                    f"Epoch {epoch} train_loss"
                )
                if "❌" in result:
                    print(f"    {result}")
                    issues_count += 1
                    errors_found = True

            # Check test MAP
            if 'test_map' in log_epoch:
                result = compare_values(
                    log_epoch['test_map'],
                    json_epoch.get('test_map'),
                    f"Epoch {epoch} test_map"
                )
                if "❌" in result:
                    print(f"    {result}")
                    issues_count += 1
                    errors_found = True

            # Check best MAP
            if 'best_map' in log_epoch:
                result = compare_values(
                    log_epoch['best_map'],
                    json_epoch.get('best_map'),
                    f"Epoch {epoch} best_map"
                )
                if "❌" in result:
                    print(f"    {result}")
                    issues_count += 1
                    errors_found = True

            # Check PR values
            if 'pr_values' in log_epoch:
                result = compare_values(
                    log_epoch['pr_values'],
                    json_epoch.get('pr_values'),
                    f"Epoch {epoch} pr_values"
                )
                if "❌" in result:
                    print(f"    {result}")
                    issues_count += 1
                    errors_found = True

        if issues_count == 0:
            print(f"  ✓ All values match!")

    return errors_found

def main():
    base_path = Path("/Users/tuanha/Work/learn/master_is/semester_3/information_retrieval/lab/VisionTransformerHashing/assets/scenario_logs")

    scenarios = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('scenario-')])

    total_errors = False
    for scenario_path in scenarios:
        if verify_scenario(scenario_path):
            total_errors = True

    print(f"\n{'='*60}")
    if total_errors:
        print("⚠️  Some values do NOT match between JSON and log files!")
    else:
        print("✓ All values match! JSON files are correct.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
