with open('accuracy_YOLO.lg','r') as f: lines = [l.strip().split(',') for l in f][1:]
for line in sorted(lines, key = lambda x: x[2]):
    print(line)

