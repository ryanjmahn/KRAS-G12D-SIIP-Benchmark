"""Quick check: does PocketMiner output exist for both anchors?"""
import os
for name in ['5US4_H','7RPZ_H']:
    for path in [f'ae-pocketminer/results/pocketminer/{name}-preds.npy',
                 f'results/pocketminer/{name}-preds.npy']:
        if os.path.exists(path):
            print(f"{name}: FOUND at {path}")
            break
    else:
        print(f"{name}: NOT FOUND - need to run PocketMiner on it")