# Using readlines()
file1 = open("requirements.txt", "r")
lines = file1.readlines()
count = 0

# Strips the newline character
res = []

for line in lines:
    clean_line = line.split("=")

    if "=" not in line:
        res.append(line)
        continue

    if len(clean_line) > 2:
        clean_line.pop(-1)

    if '\n' not in clean_line[1]:
        res.append(clean_line[0] + '=' + clean_line[1] + '\n')
    else:
        res.append(clean_line[0] + '=' + clean_line[1])


# writing to file
file1 = open('clean_requirement.txt', 'w')
file1.writelines(res)
file1.close()
  
    




