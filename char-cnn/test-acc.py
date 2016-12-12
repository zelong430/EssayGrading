import glob, os, csv

os.chdir(".")

results = []
max_test_params = []
max_test_acc = 0
for file in glob.glob("*.txt"):
    params = file.split('.')[0].split('-')[2:]
    accs = []

    with open(file, "r") as f:
    	accs.append(f.readline().strip())
    	accs.append(f.readline().strip())

	if float(accs[1]) > max_test_acc:
		max_test_params = params
		max_test_acc = float(accs[1])

    params += accs
    results.append(params)

print(max_test_params)
print(max_test_acc)

with open("test-accs.csv","w") as f:
	writer = csv.writer(f, delimiter = " ")
	for line in results:
		writer.writerow(line)