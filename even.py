f = open("AGBIG_annotation.out")

f = f.readlines()

f = [i.split() for i in f]

one = [j for j in f if j[0] == "no"]
two = [j for j in f if j[0] == "yes"]

print(len(one))
print(len(two))

p = 2500
save = 0
while(p > 0):
	for i in range(save, len(f)):
		save = i
		if(f[i][0] == "yes"):
			f.remove(f[i])
			p = p - 1
			break


one = [j for j in f if j[0] == "no"]
two = [j for j in f if j[0] == "yes"]

print(len(one))
print(len(two))

out = open("a_lil_more.out", "w")
for j in f:
	for l in j:
		out.write(l + " ")
	out.write("\n")
