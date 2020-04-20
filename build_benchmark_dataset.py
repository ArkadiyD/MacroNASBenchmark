import argparse
import json
import os
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='', required=True, help='Path to experiment folder')
args = parser.parse_args()

files = os.listdir(args.dir+'/evaluations')
print ('Reading %d evaluation files from %s' % (len(files), args.dir + '/dataset.json'))
results = {}
for _, file in tqdm.tqdm(enumerate(files)):
    if 'D' not in file and 'supernet' not in file:
        #print(file)
        model = file.split('_')[1].replace('.json','')
        data = json.load(open(os.path.join(args.dir + '/evaluations', file), 'r'))
        #print(data)
        del data['test_loss']
        del data['val_loss']
        results[model] = data 
print('Saving final dataset to', args.dir + '/dataset.json')
json.dump(results, open(args.dir + '/dataset.json', 'w'))
            
