import json
with open(r'C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje\light_script.json') as f:
    data = json.load(f)

print("=== 1:00 - 1:30 arasi ===")
for frame in data[120:150]:
    z1, z2, z3 = frame["zone1"], frame["zone2"], frame["zone3"]
    print(f"t={frame['timestamp']}  z1={z1}  z2={z2}  z3={z3}")
