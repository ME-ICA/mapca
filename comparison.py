import pandas as pd
import os
import json
#import subprocess

repoid="ds000258"
#repourl="git@github.com:OpenNeuroDatasets/"
#wdir="/bcbl/home/home_g-m/llecca/Scripts/gift/dev"
wdir=os.getcwd()

#subprocess.run(["cd",wdir])
#os.system(["cd", wdir])

os.system("datalad install git@github.com:OpenNeuroDatasets/"+repoid+".git")
repo=os.path.join(wdir,repoid)

subjects=[]

for file in os.listdir(repo):
    d = os.path.join(repo,file)
    if os.path.isdir(d) and "sub-" in d:
        subjects.append(os.path.basename(d))

json_files=[]
echo_times=[]

for sbj in subjects:
    for file in os.listdir(os.path.join(repo,sbj,'func')):
        if file.endswith('.json'):
           f=open(os.path.join(repo,sbj,'func',file))
           data=json.load(f)
           echo_times.append(data["EchoTime"])
           json_files.append(file)
           f.close()

#df_echo=pd.DataFrame(list(zip(json_files,echo_times)),columns=['file','echo time'])

df_echo = pd.DataFrame({'file': json_files, 'echo time': echo_times})

#write.csv(df_echo,os.path.join(wdir,'echo_times_'+repoid+'.csv'),row.names=FALSE)
df_echo.to_csv(os.path.join(wdir,'echo_times_'+repoid+'.csv'),index=False)


# here run matlab script







#df=pd.read_csv(os.path.join(wdir,'participants.tsv'),sep='\t')
#subjects=df['participant_id']

#for sbj in subjects:
#    print(os.path.join(sbj,'func'))
#    for jsonfile in os.listdir(os.path.join(wdir,sbj,'func')):
#        if jsonfile.endswith('.json'):
#            f=open(jsonfile)
#            data=json.load(f)
#            print(data['EchoTime'])
#            f.close()


    #datalad get os.path.join(sbj,'func')
    #datalad drop os.path.join(sbj,'func')
