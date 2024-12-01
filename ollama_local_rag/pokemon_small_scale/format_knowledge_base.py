import csv 

data = []
with open('pokemon.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            # ONLY CARE ABOUT POKEDEX NUMBER
            data.append({'id': row[2].zfill(3), 'name': row[1]})
            line_count += 1
    print(f'Processed {line_count} lines.')

# write data
for i, _ in enumerate(data):
    with open(f'knowledge_base/data/{i + 1}.txt', 'w') as f:
        f.write(f"Pokedex Number {_['id']}: Name: {_['name'].capitalize()}")