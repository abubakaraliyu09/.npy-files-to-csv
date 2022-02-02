import pandas as pd
import glob
import numpy as np
import os

ids=[]
features=[]

for f in glob.glob('data1\\*.npy'):
    names=os.path.basename(f)
    name=names.strip('_AUDIO_audio_features.npy')
    ids.append(name)
    #print(name)
    features.append(np.load(f))
x=pd.DataFrame(np.concatenate(features))#.to_csv('Features.csv',index=False)
y=pd.DataFrame(ids)#.to_csv('IDS.csv',index=False)
horizontal_stack = pd.concat([y, x], axis=1)
print(horizontal_stack)
horizontal_stack.to_csv('Concat.csv',index=False)
