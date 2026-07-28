# -*- coding: utf-8 -*-
"""
Label statistics script: per-class instance count + area scale, English names, no background.
"""
import os, csv, json, sys
from collections import Counter, defaultdict
import numpy as np

CSV_PATH = r"D:\CD\0617final\classes.csv"
SCENES = ["Airports", "Ports", "Urban-Rural Areas"]

TARGET_EN = [
    "Farmland", "Runway", "Taxiway", "Apron", "Bridge", "Highway",
    "Building", "Shelter", "Tower", "Pier", "Dock", "Tank",
    "Aircraft", "Vessel", "Crater", "VehicleRevet"
]
STATE_EN = ["NoChange", "Damaged", "Reduced", "Added", "Extended", "Replaced"]

def load_classes(csv_path):
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            tid = int(row['train_id'])
            mapping[tid] = {"target_zh": row['target_zh'], "state": row['state']}
    return mapping

def stats_from_instances(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    combo_counter = Counter()
    combo_areas = defaultdict(list)
    for img_info in data.values():
        for inst in img_info['instances']:
            ti, si = inst['target_idx'], inst['state_idx']
            combo_counter[(ti, si)] += 1
            combo_areas[(ti, si)].append(inst['area'])
    return combo_counter, combo_areas, len(data)

def main():
    class_map = load_classes(CSV_PATH)
    all_rows = []

    for scene in SCENES:
        json_path = os.path.join(r"D:\CD\0617final", scene, "instances.json")
        if not os.path.exists(json_path):
            continue

        combo_counter, combo_areas, n_imgs = stats_from_instances(json_path)
        total_inst = sum(combo_counter.values())

        print(f"\n{'='*95}")
        print(f"{scene}  ({n_imgs} images, {total_inst} instances)")
        print(f"{'='*95}")
        print(f"{'Target':<12} {'State':<10} {'Count':>7} {'%':>6} {'AreaMin':>8} {'AreaMed':>8} {'AreaMax':>8} {'AreaMean':>9}")
        print("-"*75)

        for ti in range(len(TARGET_EN)):
            pairs = [(ti, si) for si in range(len(STATE_EN)) if combo_counter.get((ti, si), 0) > 0]
            if not pairs:
                continue
            for si in range(len(STATE_EN)):
                cnt = combo_counter.get((ti, si), 0)
                if cnt == 0:
                    continue
                areas = combo_areas[(ti, si)]
                pct = cnt / total_inst * 100
                a_min = min(areas)
                a_med = int(np.median(areas))
                a_max = max(areas)
                a_mean = int(np.mean(areas))
                print(f"{TARGET_EN[ti]:<12} {STATE_EN[si]:<10} {cnt:>7} {pct:>5.1f}% {a_min:>8} {a_med:>8} {a_max:>8} {a_mean:>9}")
                all_rows.append([scene, TARGET_EN[ti], STATE_EN[si], cnt, f"{pct:.1f}%",
                                 a_min, a_med, a_max, a_mean])

        # Per-target summary
        print(f"\n  Per-target summary:")
        target_totals = Counter()
        target_areas = defaultdict(list)
        for (ti, si), cnt in combo_counter.items():
            target_totals[ti] += cnt
            target_areas[ti].extend(combo_areas[(ti, si)])
        for ti, cnt in sorted(target_totals.items(), key=lambda x: -x[1]):
            areas = target_areas[ti]
            pct = cnt / total_inst * 100
            a_med = int(np.median(areas))
            print(f"    {TARGET_EN[ti]:<12} {cnt:>7} ({pct:.1f}%)  median_area={a_med}")

    # Save CSV
    csv_out = r"D:\CD\0617final\label_statistics.csv"
    with open(csv_out, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "target", "state", "count", "pct",
                         "area_min", "area_median", "area_max", "area_mean"])
        writer.writerows(all_rows)
    print(f"\nCSV saved: {csv_out}")

if __name__ == "__main__":
    main()
