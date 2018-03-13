import json



with open('./4_gps.json') as data_file:    
    data = json.load(data_file)
f = open('./gpsdata.txt', 'w')



i = 0
while True:
	try:
		i += 1
		accuracy = data["locations"][i]["accuracy"]
		longitude = data["locations"][i]["longitudeE7"]
		latitude = data["locations"][i]["latitudeE7"]
		timestamp = data["locations"][i]["timestampMs"]

		line = str(accuracy) + '\t' + str(longitude) + '\t' + str(latitude) + '\t' + str(timestamp) + '\n'
		f.write(line)
	except:
		break


f.close()




