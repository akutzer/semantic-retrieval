#from transformers import AutoTokenizer

#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#sequence = "In a hole in the ground there lived a hobbit."
#print(tokenizer(sequence))

import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/tinyroberta-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)



question = 'When was Harry born?'
'''QA_inputs = [
    {    'question': question,
    'context': 'Harry James[58] Potter (b. 31 July 1980)[1] was an English half-blood[2] wizard, and one of the most famous wizards of modern times. '
               'The only child and son of James and Lily Potter (née Evans), Harry\'s birth was overshadowed by a prophecy, naming either himself or Neville Longbottom as the'
               ' one with the power to vanquish Lord Voldemort, the most powerful and feared Dark wizard in the world. After half of the prophecy was reported to Voldemort, courtesy '
               'of Severus Snape, Harry was chosen as the target due to his many similarities with the Dark Lord. In turn, this caused the Potter family to go into hiding.'},
    {'question': question,
     'context': 'Voldemort made his first attempt to circumvent the prophecy when Harry was a year and three months old. During this attempt, he murdered Harry\'s parents as they'
                ' tried to protect him, but this unsuccessful attempt to kill Harry led to Voldemort\'s first downfall. This downfall marked the end of the First Wizarding War, and to'
                ' Harry henceforth being known as "The Boy Who Lived",[5] as he was the only known survivor of the Killing Curse due to being magically protected by his mother\'s loving sacrifice.'
                'In accordance with the terms of the prophecy, this attempt on his life also established him, not Neville, as Voldemort\'s nemesis.'},
    {'question': question,
     'context': 'Shortly before Harry\'s eleventh birthday, there were several attempts from Hogwarts School of Witchcraft and Wizardry to send him a letter inviting him to not only come to Hogwarts,'
                ' but also to explain his magical heritage. Though Vernon Dursley ultimately made the decision to leave their residence at 4 Privet Drive temporarily to evade the letters, his attempts were '
                'ultimately for naught. On his eleventh birthday, Harry learned from Rubeus Hagrid that he was a wizard.[60] Rubeus Hagrid took Harry from the Dursley family the next day to Diagon Alley. '
                'This is where Harry withdrew some of his money from Gringotts Wizarding Bank, where Hagrid bought Harry\'s snow-white owl, Hedwig. While he was in Diagon Alley, Harry also got his school '
                'supplies (i.e., books from Flourish and Blotts, his robes from Madam Malkin\'s Robes for All Occasions, and his potion equipment from Slug & Jiggers Apothecary). Harry began attending Hogwarts '
                'in 1991. The Sorting Hat told Harry that he would do well in Slytherin House, but Harry pleaded "not Slytherin". The Hat heeded this plea and sorted the young wizard into Gryffindor House.[56] '
                'At school, Harry became best friends with Ron Weasley and Hermione Granger. He was also the youngest Seeker in a century, making the house team in his first year when Minerva McGonagall '
                'introduced him to Oliver Wood, Gryffindor\'s Quidditch captain at the time. He later became the captain of the Gryffindor Quidditch Team in his sixth year, winning two Quidditch Cups.[61] '
                'While in school, Harry also demonstrated an extraordinary talent for Defence Against the Dark Arts.'},
    {'question': question,
     'context': 'He became even better-known in his early years for protecting the Philosopher\'s Stone from Voldemort, saving Ron\'s sister Ginny Weasley, solving the mystery of the Chamber of Secrets, '
                'slaying Salazar Slytherin\'s basilisk, and learning how to conjure a corporeal Patronus at the age of thirteen, which took the form of a stag. In his fourth year, Harry won the '
                'Triwizard Tournament, although the person had to be seventeen to enter, and Harry was only fourteen. The competition ended with the tragic death of Cedric Diggory and the return of '
                'Lord Voldemort. During the next school year, in defiance of Dolores Umbridge\'s and the Ministry of Magic\'s strict regime against the teaching of Defence Against the Dark Arts and the continued '
                'introduction of the new Educational Decree, Ronald Weasley and Hermione Granger first had the idea of creating an illegal Defence Against the Dark Arts group called Dumbledore\'s Army. Harry was'
                ' initially reluctant to the idea, but he gradually came to like it and eventually agreed to start the group. He also fought in the Battle of the Department of Mysteries, during which he lost his godfather, '
                'Sirius Black, and the prophecy was destroyed.'},
    {'question': question,
     'context': 'One consequence of Lily\'s sacrifice was that her orphaned son had to be raised by her only remaining blood relative, his Muggle aunt Petunia Dursley, '
                'and her husband, uncle Vernon Dursley. While in their care, he would be protected from Lord Voldemort due to the Bond of Blood charm that Albus Dumbledore '
                'placed upon him.[59] This powerful charm would protect him until he either came of age, or no longer called his aunt\'s house home. Due to Petunia\'s resentment '
                'of her sister and her magical abilities, Harry grew up abused and neglected.'}
]


res = []
best = {'score': -1, 'start': 0, 'end': 0, 'answer': ''}

for i in range (0, len(QA_inputs)):
    answer = nlp(QA_inputs[i])
    res.append(answer)
    #print(answer['score'])
    if answer['score'] > best['score']:

        best = answer
        paragraph_index = i


print("all answers per paragraph: ", res)
print("best result: ", best)
print("in paragraph:", paragraph_index+1)'''

#whole harry wiki
df = pd.read_pickle("C:/Daten/Florian/Dev/Python/information_retrieval1/res/harrypotter_pages_current.pickle")

question = 'what was owned by an old witch?'
res = []
best = {'score': -1, 'start': 0, 'end': 0, 'answer': ''}
number_of_paragraphs_to_search = 100
for i in range(0, 100):
    dict = {'question': question,
     'context': df["text"].iloc[i]}
    answer = nlp(dict)
    if answer['score'] > best['score']:
        best = answer
        paragraph_index = i


print("all answers per paragraph: ", res)
print("best result: ", best)
print("paragraph_index:", paragraph_index)
print("paragraph:", df["text"].iloc[paragraph_index])



# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)