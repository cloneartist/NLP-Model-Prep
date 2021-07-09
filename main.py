
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm
from spacy.training import Example

TRAIN_DATA = [('The dark web is an enabler for the circulation of illegal weapons such as bombs.', {'entities': [(58, 65, 'WEAPON'), (74, 79, 'WEAPON')]}), ('Dark web markets that facilitate the sale of firearms and explosives.', {'entities': [(45, 53, 'WEAPON'), (58, 68, 'WEAPON')]}), ('Pistols were the most commonly listed firearm.', {'entities': [(0, 7, 'WEAPON'), (38, 45, 'WEAPON')]}), ('Followed by rifles and sub machineguns.', {'entities': [(27, 38, 'WEAPON'), (12, 18, 'WEAPON')]}), ('To the full manufacture of home made guns and explosives.', {'entities': [(46, 56, 'WEAPON'), (37, 41, 'WEAPON')]}), ('Also include models that can be turned into fully working firearms and guns through 3D printing.', {'entities': [(84, 86, 'CARDINAL'), (58, 66, 'WEAPON'), (71, 75, 'WEAPON')]}), ('Every month there could be up to 136 untraced firearms and guns or associated products in the real world.', {'entities': [(0, 11, 'DATE'), (33, 36, 'CARDINAL'), (46, 54, 'WEAPON'), (59, 63, 'WEAPON')]}), ('\n', {'entities': []}), ('What I found most surprising was that most of what we saw wasn’t rifles of military-grade weapons,Holt says.', {'entities': [(98, 102, 'PERSON'), (90, 97, 'WEAPON'), (65, 71, 'WEAPON')]}), ('\n', {'entities': []}), ('Instead of exotic or rare firearms, we saw handguns—the kinds of weapons someone in the US could buy from stores or vendors with a license. ', {'entities': [(88, 90, 'GPE'), (65, 72, 'WEAPON'), (26, 34, 'WEAPON'), (43, 51, 'WEAPON')]}), ('Additionally, the price points of these guns weren’t drastically different than what you’d find if you were buying legally. ', {'entities': [(40, 44, 'WEAPON')]}), ('Sixty-four percent of the products advertised were handguns, 17 percent were semi-automatic long guns, and fully automatic long guns were 4 percent.', {'entities': [(0, 18, 'PERCENT'), (61, 71, 'PERCENT'), (138, 147, 'PERCENT'), (97, 101, 'WEAPON'), (128, 132, 'WEAPON'), (51, 59, 'WEAPON')]}), ('There are many reasons buyers could turn to the dark web to purchase a weapon, Holt explains. ', {'entities': [(79, 83, 'PERSON'), (71, 77, 'WEAPON')]}), ('One example would be a buyer who can’t legally obtain a firearm; another explanation would be that the buyer lives in a country with stricter gun laws.', {'entities': [(0, 3, 'CARDINAL'), (56, 63, 'WEAPON'), (142, 145, 'WEAPON')]}), ('Regardless, Holt says that because the dark web allows for total anonymity, it supports his theory that the dark web buyers are those who wouldn’t be able to purchase a firearm legally.', {'entities': [(12, 16, 'PERSON'), (169, 176, 'WEAPON')]})]

model = None
n_iter=100

if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')
    print("Created blank model")

#set up the pipeline

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe('ner', last=True)
else:
    ner = nlp.get_pipe('ner')

for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])


other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            doc=nlp.make_doc(text)
            example= Example.from_dict(doc, annotations)
            nlp.update(
                [example],

                drop=0.5,
                sgd=optimizer,
                losses=losses)
        print(losses)

for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

if output_dir is not None:
    output_dir = Path("D:\\Legion\\nlptrain\\Model")
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

    
# Code to Load Model
# import spacy.displacy as displacy
# import spacy
# nlp = spacy.load('D:\\Legion\\nlptrain\\Model')
# inp=input()
# doc = nlp(inp)

# for ent in doc.ents:
#     print(ent.text, ent.label_)
# displacy.serve(doc, style='ent')
    


