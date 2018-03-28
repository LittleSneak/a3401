import os
import numpy as np
import string

dataDir = '/u/cs401/A3/data/'
#dataDir = 'C:/Users/Admin/401a3/a3401/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    #Initialize the array with an extra row and column for infs
    arr = np.zeros((len(r) + 1, len(h) + 1))
    
    #Main double loop which iterates through
    #The hypotheses for every word in the ref
    for i in range(0, len(arr)):
        for j in range(0, len(arr[0])):
            #Position [0][0] is 0 and the top row and left column are all inf
            if i == 0 or j == 0:
                if i == 0 and j == 0:
                    arr[i][j] = 0
                elif i == 0:
                    arr[i][j] = j
                elif j == 0:
                    arr[i][j] = i
            else:
                #Check if we need to add 1 or not to sub
                add = 1
                if(i > 0 and j > 0 and r[i - 1] == h[j - 1]):
                    add = 0
                    
                arr[i][j] = min(arr[i-1][j] + 1, arr[i-1][j-1] + add, arr[i][j-1] + 1)
                
    #Work backwards to find the number of subs/ins/dels
    sid = [0, 0, 0]
    i = len(arr) - 1
    j = len(arr[0]) - 1
    while i != 0 or j != 0:
        #substitution
        if(i > 0 and j > 0 and (arr[i][j] - arr[i-1][j-1] == 1)):
            i = i - 1
            j = j - 1
            sid[0] = sid[0] + 1
        #Insertion
        elif(j > 0 and (arr[i][j] - arr[i][j-1] == 1)):
            j = j - 1
            sid[1] = sid[1] + 1
        #Deletion
        elif(i > 0 and (arr[i][j] - arr[i-1][j] == 1)):
            i = i - 1
            sid[2] = sid[2] + 1
        #Matching
        else:
            i = i - 1
            j = j - 1
    ret = []
    
    #Avoid division by 0 if ref sentence is empty
    if(len(arr) - 1 > 0):
        ret.append(arr[len(arr) - 1][len(arr[0]) - 1] / (len(arr) - 1))
    else:
        ret.append(float('inf'))
        
    ret.append(sid[0])
    ret.append(sid[1])
    ret.append(sid[2])
    return ret

def preprocess(s):
    """
    Remove all punctuation from an input sentence and convert
    to lower case.
    """
    newString = ""
    for letter in s:
        if not letter in string.punctuation:
            newString = newString + letter
        if letter == "[" or letter == "]":
            nreString = newString + letter
    return newString.lower()

if __name__ == "__main__":
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            #Location of text files for this speaker
            fileDir = dataDir + speaker + "/"
            ref_lines = []
            google_lines = []
            kaldi_lines = []
            
            #Open each file and add the preprocessed lines to the lists
            with open(fileDir + "transcripts.txt", 'r') as f:
                for line in f:
                    ref_lines.append(preprocess(line))
            with open(fileDir + "transcripts.Google.txt", 'r') as f:
                for line in f:
                    google_lines.append(preprocess(line))
            with open(fileDir + "transcripts.Kaldi.txt", 'r') as f:
                for line in f:
                    kaldi_lines.append(preprocess(line))
            #Do the printouts
            for i in range(0, len(ref_lines)):
                scores = Levenshtein(ref_lines[i].split(), google_lines[i].split())
                print("[",speaker ,"]", " [", "Google", "]", " [",i , "]", " [", scores[0], "]", "S:[", scores[1], "], I:[", scores[2], "], D:[", scores[3], "]")
                scores = Levenshtein(ref_lines[i].split(), kaldi_lines[i].split())
                print("[",speaker ,"]", " [", "Kaldi", "]", " [",i , "]", " [", scores[0], "]", "S:[", scores[1], "], I:[", scores[2], "], D:[", scores[3], "]")                
            
