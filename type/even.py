f = open("AGBIG.out")

f = f.readlines()

f = [i.split() for i in f]

fl = [j for j in f if j[0] == "Facts/Logic"]
pt = [j for j in f if j[0] == "Positive"]
nt = [j for j in f if j[0] == "Negative"]
a = [j for j in f if j[0] == "Affiliation"]
h = [j for j in f if j[0] == "Humor"]
w = [j for j in f if j[0] == "Warning"]

#print(len(fl))
#print(len(pt))
#print(len(nt))
#print(len(a))
#print(len(h))
#print(len(w))

p = 241
save = 0
while(p > 0):
	for i in range(save, len(f)):
		save = i
		if(f[i][0] == "Positive"):
			f.remove(f[i])
			p = p - 1
			break


one = [j for j in f if j[0] == "Positive"]
two = [j for j in f if j[0] == "Negative"]

print(len(one))
print(len(two))

out = open("a_lil_morenp.out", "w")
for j in f:
	for l in j:
		out.write(l + " ")
	out.write("\n")

