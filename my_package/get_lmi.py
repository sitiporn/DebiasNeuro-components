import pickle
with open('top_lmi_sent1.pickle', 'rb') as handle:
    top_lmi_sent1 = pickle.load(handle)
print(top_lmi_sent1)

with open('top_lmi_sent2.pickle', 'rb') as handle:
    top_lmi_sent2 = pickle.load(handle)
print(top_lmi_sent2)
