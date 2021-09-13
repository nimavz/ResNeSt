import glob
import os

path = r"dataset_n/fake/"
# search text files starting with the word "sales"
pattern = path + "hard" + "*.jpg" ## or easy, or mid, or real

# List of the files that match the pattern
result = glob.glob(pattern)

# Iterating the list with the count
count = 1
for file_name in result:
    old_name = file_name
    new_name = path + 'fake_3_' + str(count) + ".jpg"
    os.rename(old_name, new_name)
    count = count + 1

# printing all revenue txt files
res = glob.glob(path + "fake_3_" + "*.jpg")
for name in res:
    print(name)