import csv 

with open('AGBIG.csv', newline='') as csvfile:
    file_array = list(csv.reader(csvfile))

f = open("AGBIGnp.out", "w")
#types = ["Facts/Logic", "Positive tone", "Negative tone", "Affiliation", "Humor", "Warning"]
types = ["Negative tone", "Positive tone"]
for i in file_array:
	if(i[2] in types):
		if(i[2] == "Negative tone"):
			i[2] = "Negative"
		if(i[2] == "Positive tone"):
			i[2] = "Positive"
		f.write(i[2] + " " + i[0] + "\n")	
