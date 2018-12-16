from pyrouge import Rouge155
import sys
import pickle
model = sys.argv[1]
target = sys.argv[2]
outputdir = sys.argv[3]
r = Rouge155('/home/lgpu0237/ROUGE-1.5.5')
r.system_dir = target
r.model_dir = model
# 002691_decoded.txt
r.system_filename_pattern = '(\d+)_reference.txt'
r.model_filename_pattern = '#ID#_decoded.txt'

output = r.convert_and_evaluate()
print(output)
text_file = open(outputdir+"/pyrouge.txt", "w")
text_file.write(output)
text_file.close()
output_dict = r.output_to_dict(output)

pickle.dump(output_dict, open(outputdir+'/pyrouge_dict.score', 'wb'))
