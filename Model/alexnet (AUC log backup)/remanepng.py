import os
	
files = [ file for file in os.listdir(os.getcwd()) if file.endswith(".PNG")]

for file in files:
	print(file)
	os.rename(file,file.replace("_", "-").replace("0.", "0"))
