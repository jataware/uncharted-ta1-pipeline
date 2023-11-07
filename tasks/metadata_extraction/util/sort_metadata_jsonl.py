import jsonlines
import sys
import json

# Define the input and output file paths
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

# Load records from the input file
json_objs = []
with open(input_file_path, "r") as reader:
    for line in reader:
        json_objs.append(json.loads(line))

# Sort records based on the GEO_ key
sorted_obj_list = sorted(json_objs, key=lambda x: int(next(iter(x))))

# Save the sorted records to the output file
with jsonlines.open(output_file_path, "w") as writer:
    for record in sorted_obj_list:
        writer.write(record)
