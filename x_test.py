import json

print('running')
with open('/data/x.json', 'r') as file:
    data = json.load(file)
    file.close()

print(len(data))