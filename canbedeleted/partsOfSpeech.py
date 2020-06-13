#%%

import spacy   # Must be installed as spacy
import urllib.request, json

sent = """Dear Sir Graham Brady. My name is Puria Radmard, and I live in Broadheath ward. I am writing to you in regards to the recent death of Belly Mujinga, her death follows after an assault was carried out on 21st March in which her and colleagues were spat at and coughed on during their shift at London Victoria Station. 

As I am sure you are aware, it was revealed on 29th May that The British Transport Police ruled that they believed there was no link between the act of assault and her death and stated that ‘no further action will be taken’, closing the case. In spite of this Ms. Mujinga’s passing on 5th April comes just two weeks after the assault had taken place and after several days having been admitted to hospital and testing positive for COVID-19.
"""
nlp = spacy.load("en_core_web_sm")
doc = nlp(sent)

final = ""

for i, token in enumerate(doc):
    print("original:", token.orth, token.orth_)
    #print("lowercased:", token.lower, token.lower_)
    #print("position:", token.pos, token.pos_)
    #print("log probability:", token.prob)
    #print("Brown cluster id:", token.cluster)
    #print("— — — — — — — — — — — — — — — — — — — —")
    
    if token.pos_ == "PUNCT" and token.orth_ != "‘": final = final[:-1]
    final += token.orth_ + " "
    if token.orth_ =="‘": final = final[:-1]

    if token.pos_ == "ADJ":
        with urllib.request.urlopen("http://api.datamuse.com/words?ml="+token.orth_) as url:
            data = json.loads(url.read().decode())
            print(data[0])
        


#print(final)
# %%
print(final)

# %%
