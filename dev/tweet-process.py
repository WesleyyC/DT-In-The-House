import csv
import re

ifile  = open('DT-tweet-raw.txt', "rb")
reader = csv.reader(ifile)

wfile = open('DT-tweet.txt', 'w')

for row in reader:
	text = row[0]
	text = text.replace("\"","")
	text = text.strip()
	text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
	if len(text)>0 and text[0]!='@':
		wfile.write(text)

wfile.close()
ifile.close()
