import csv
import json
from collections import OrderedDict, Counter

filename = 'dataset-27-5-2019-restructured.json'
room_number = 'room1'
output_filename = 'dataset_' + room_number + '.csv'
dates = list()

with open(filename) as json_file:
    x = json.load(json_file, object_pairs_hook=OrderedDict)
    print("Creating", output_filename, "file...")
    f = csv.writer(open(output_filename, "w", newline=''))
    f.writerow(["date", "time", 'appID', 'power(W)', "occupancy"])

    power = x["rooms"][room_number]["appliance_data"]["app_1"] # Get all values
    appid = list(x["rooms"][room_number]["appliance_data"].keys())[0]

    for index, power_item in enumerate(power.items()): # Cycle through each date
        date = power_item[0]
        dates.append(date)
        occupancy = x["rooms"][room_number]["occupancy"]
        if date in occupancy.keys():
            occupancy_today = occupancy[date]
            for index, item in enumerate(power_item[1].items()): # Cycle through each item # Cycle through each row in date
                time = item[0]
                power_value = item[1]
                occupancy_value = list(list(occupancy.values())[0].values())[index]
                f.writerow([date, time, appid, power_value, occupancy_value])
                print("Row", index, "written.")

    print("Dataset dates", dates, ", total of", len(dates))