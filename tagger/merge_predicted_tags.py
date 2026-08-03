from collections import Counter
import glob
import json
import sys


def most_frequent(List):
    occurence_count = Counter(List)
    if len(occurence_count) == 1:
        return [occurence_count.most_common(1)[0][0]]

    # in case of ties, return the two most frequent elements
    if occurence_count.most_common(2)[0][1] == occurence_count.most_common(2)[1][1]:
        return [occurence_count.most_common(2)[0][0], occurence_count.most_common(2)[1][0]]

    # no tie but more than one tag: return most freq element
    return [occurence_count.most_common(2)[0][0]]




def write_to_file(dic, outfile):
    with open(outfile, 'w') as out:
        for i in range(len(dic[1]['tok'])):
            if dic[1]['tid'][i] == 1:
                out.write('\n')
            line = str(dic[1]['sid'][i]) + '\t' + str(dic[1]['tid'][i]) + '\t' + dic[1]['tok'][i] + '\t' + dic[1]['dta'][i] + '\t'
            pos = []
            for j in range(1, 6):
                pos.append(dic[j]['pos'][i])
            line += '\t'.join(pos)
            if dic[1]['dta'][i] not in pos or len(list(set(pos))) > 1:
                line += '\tMISMATCH\n'
            else:
                line += '\t_\n'
            out.write(line)

            
def get_string(i):
    string = ''
    for j in range(0, 2-i):
        string += '\t_'
    return string    


"""
We only want two pos alternatives. Take the set of dta + pos1 + pos2 and display
two alternatives, marked by MISMATCH
"""
def write_majority_vote_to_file(dic, outfile):
    with open(outfile, 'w') as out:
        out.write("SID\tTID\tTOKEN\tPOS1\tPOS2\n")  
        for i in range(len(dic[1]['tok'])):
            if dic[1]['tid'][i] == (1):
                out.write('\n')
            line = str(dic[1]['sid'][i]) + '\t' + str(dic[1]['tid'][i]) + '\t' + dic[1]['tok'][i] + '\t' + dic[1]['dta'][i]
            pos = []
            for j in range(1, 6):
                pos.append(dic[j]['pos'][i])

            # get most frequent tag(s)
            # in case of ties, return the 2 mf tags
            pos = most_frequent(pos)
 
            if dic[1]['dta'][i] not in pos or len(pos) > 1:
                if pos[0] == dic[1]['dta'][i]:
                    line += '\t' + pos[1] + '\n' 
                else:
                    line += '\t' + pos[0] + '\n'  
            else:
                line += '\t' + pos[0] + '\n'  
            out.write(line)
            
            


def read_preds(i, dic, predfile): 
    with open(predfile, 'r') as inf:
        for line in inf:
            if line.startswith('\t'):
                pass
            else:
                elms = line.strip().split('\t')
                if i == 1:
                    dic[i]['sid'].append(str(int(elms[0])+1))
                    dic[i]['tid'].append(elms[1])
                    dic[i]['tok'].append(elms[2])
                    dic[i]['dta'].append(elms[3])
                dic[i]['pos'].append(elms[4])




indir = 'tagger_input/'
pred_dir = 'tagger_output/run'

infiles = glob.glob(indir+'*.tsv')

for f in infiles:
    print(f)
    dic = {i:{'pos':[]} for i in range(1, 6)}
    dic[1]['sid'] = []
    dic[1]['tid'] = []
    dic[1]['tok'] = []
    dic[1]['dta'] = []

    for i in range(1, 6):
        pred_file = f.replace(indir, pred_dir+str(i)+'/') 
        read_preds(i, dic, pred_file)

    outfile = pred_file.replace('/run'+str(i),'').replace('.tsv', '.merged.tsv')
    write_to_file(dic, outfile)
    write_majority_vote_to_file(dic, outfile.replace('merged', 'merged_mv'))
