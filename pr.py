from pyrouge import Rouge155
import sys
import pickle
reload(sys)
sys.setdefaultencoding('utf8')

base = sys.argv[1]

model = base + sys.argv[2]
target = base + sys.argv[3]

r = Rouge155('/home/lgpu0237/ROUGE-1.5.5')
r.system_dir = target
r.model_dir = model
# 002691_decoded.txt
r.system_filename_pattern = '(\d+)_ref'
r.model_filename_pattern = '#ID#_gen'

output = r.convert_and_evaluate()
print(output)
text_file = open(base+"/pyrouge.txt", "w")
text_file.write(output)
text_file.close()
output_dict = r.output_to_dict(output)

pickle.dump(output_dict, open(base + '/pyrouge_dict.score', 'wb'))
