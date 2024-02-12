import os
import datetime

def log(logdir, msg):
    f = open(os.path.join(logdir, 'log'), 'a')
    dateStr=datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d:%H%M%S') 
    #dateStr=datetime.datetime.now() 
    msg = insert_environment_variables_into_path( msg )
    string = dateStr + " * " + msg
    print(string)
    f.write(string + "\n")
    f.close()

def insert_environment_variables_into_path( msg ):
    toks = msg.split(' ')
    toks_mod = []
    for token in toks:
        if os.getenv('SAMPLESDIR') in token:
            token = token.replace(os.getenv('SAMPLESDIR'),"$SAMPLESDIR")
        if os.getenv('TEMPLATES') in token:
            token = token.replace(os.getenv('TEMPLATES'),"$TE")
        if os.getenv('SCRATCHDIR') in token:
            token = token.replace(os.getenv('SCRATCHDIR'),"$SCRATCHDIR")
        toks_mod.append(token)
    msg_mod = ' '.join(toks_mod)
    return msg_mod
