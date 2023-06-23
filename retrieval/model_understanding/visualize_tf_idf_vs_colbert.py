from retrieval.models import ColBERTInference, load_colbert_and_tokenizer
from retrieval.model_understanding.visualize_similarity import visualize
from retrieval.model_understanding.tf_idf_vs_colbert import colbert_vs_tf_idf


if __name__ == "__main__":
    dicts = colbert_vs_tf_idf(size_datasets_good=100, size_datasets_bad=100, testing_max_count=100_000_000,
                             K_good=400, return_size=20)
    # dicts = []
    # dicts.append({('How can Aerondight be obtained? (Answered)', '[Aerondight returns and can be imported into ] Aerondight is a silver sword that can be regained by completing the quest There Can Be Only One.'): (0, 0), ('What is the known inhabitant of Venin Rocks?', "Venin Rocks is a small sub-region within the Sansretour Valley of Toussaint, located west of the Trading Post. In , it's home to the last known Silver Basilisk."): (0, 0), ('How are quarter-elves perceived by those pickiest about race?', 'Quarter-elf, or the rarely used term quadroon, are terms used on the Continent to refer to one who has a quarter elven blood, being born of one human and one half-elf or two quarter-elves. Only those pickiest about race consider them nonhuman.'): (0, 0), ('What did Olgierd offer to pay off in order to save face for the Bilewitz family?', "After Olgierd's family fell on hard times and lost their standing in high society, he and his wife intended to give their daughter's hand in marriage to a visiting Ofieri prince named Sirvat. However, this didn't work out as the prince was cursed and Olgierd regained his family's fortune, thus giving them little choice but to accept Olgierd's proposal to marry Iris. Despite this, he and his wife openly voiced their disagreement with Iris' choice in a husband, causing family gatherings to be fraught with tension. Not helping the situation, he also tried to give Olgierd bank notes to pay off any remaining debts he had so it wouldn't look bad on the Bilewitz's."): (0, 0), ("What is the suspected reason for the anomalies? Elder Blood in the child's veins.", '[Source Child] I have learned a child is the source of the anomalies. Apparently, Elder Blood might course through the child\'s veins. Triss asked me to take the child from the hospital, but Shani doesn\'t want to let him go. I don\'t think Shani trusts Triss. "I\'ll take the source-child from the hospital and take him to Triss."'): (0, 0), ('Who was Reidrich and what was his role at Kaer Morhen?', "Reidrich was a mage at Kaer Morhen and one of the last ones who knew the mutagenic secrets to create witchers. His death, along with the sacking of Kaer Morhen, led to the School of the Wolf's decline as they were no longer able to create witchers. [Biography] TBA"): (0, 0), ('How many different forms does Cidarian gambeson come in?', 'Cidarian gambeson is a light armor in that comes in 3 different forms: [Cidarian gambeson (common)] It can be purchased from the following merchants:'): (0, 0), ('What is the name of the capital city of the Nilfgaardian province?', 'Maecht is the capital of the Nilfgaardian province bearing the same name.'): (0, 0), ('What will happen if you attack or inquire about Whoreson Junior when in the arena and casino?', "[Walkthrough] if you attack or ask about Whoreson Junior at any time while visiting the arena and casino, you will not be able to use a secret passage later. The same applies if you get Cleaver's help, although a workaround is possible for that in the casino."): (0, 0), ('What is the starting point for The Price of Peace quest?', 'The Price of Peace is a story quest in . It starts when you reach the gates of Gatberg.'): (0, 0), ('How many retail copies has the game sold to date?', '[Adaptations] On October 26, 2007, Polish game publisher CD PROJEKT RED released a PC game based on this universe, called . It was very well received and a commercial success, selling over 1.5 million retail copies to date. Two more games were later released: on May 17, 2011, and on May 19, 2015.'): (0, 0), ('Who is Korin presumed to be related to?', "Korin was a Nordling warrior who was presumably Geralt of Rivia's father."): (0, 0), ("What was Szlifierz kości supposed to do in the game's development?", "Szlifierz kości was supposed to appear during the game's development in 2013 - 2014 but beast was removed around this time."): (0, 0), ('What gear set does the Grandmaster Legendary Wolven armor belong to?', 'Grandmaster Legendary Wolven armor is a craftable medium armor and is part of the Wolf School Gear in the with the New Game + option.'): (0, 0), ('Who visited the protagonist several days after his encounter with Bonhart?', "Several days later he was visited by a faction of Stefan Skellen's gang to learn what transpired between him and Bonhart. Esterhazy proved to be stubborn though, only revealing that Bonhart had dropped in and moved to use his whistle, but before he could one of the members, Joanna Selborne, who was a psionic, used her abilities to stop him before forcing him to reveal everything that pertained to the bounty hunter."): (0, 0), ("Whose journal is Speedy Eugene's journal?", "Speedy Eugene's journal is a book in the . It belonged to Speedy Eugene and is found on his body at The Silver Salamander Inn after Geralt liberates the place."): (0, 0), ('What is the requirement for the A-ten-hut! achievement?', "A-ten-hut! is an achievement in . It requires one to unlock all of Reynard's Buildings. To complete this achievement, you must fully upgrade the armory and the engineers' drafting deck in your camp."): (0, 0), ('Is there a special name for the Aard circle of elements?', '[Notes] There are only two circles of elements and only one of them is for Aard. It has no special name.'): (0, 0), ("Who killed Iris' father? - Olgierd slammed his head against a concrete pillar.", "After Olgierd and Iris ceased loving each other, Bilewitz came to his son-in-law to nullify the marriage but they argued and Iris' father was killed when Olgierd slammed his head against a concrete pillar."): (0, 0), ('What is the role or function of Soldier Puppet within the story?', 'Soldier Puppet is acard in . It appears during the Remnants battle.'): (0, 0)})
    # dicts.append(dicts[0])
    # dicts.append(dicts[0])
    # dicts.append(dicts[0])

    CHECKPOINT_PATH = "../../data/colbertv2.0/"
    colbert, tokenizer = load_colbert_and_tokenizer(CHECKPOINT_PATH)
    inference = ColBERTInference(colbert, tokenizer)

    K = 2
    kde_heatmaps, count_heatmaps, sum_heatmaps = [], [], []

    queries = [[], [], [], []]
    passages = [[], [], [], []]

    j = 0
    with open('combined_heatmaps.html', 'w', encoding="utf-8") as f:
        for dict in dicts:
            for (query, passage) in dict.keys():
                queries[j].append(query)
                passages[j].append(passage)

                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
                kde_heatmaps.append(kde_heatmap)
                count_heatmaps.append(count_heatmap)
                sum_heatmaps.append(sum_heatmap)

            if j == 0:
                f.write('<h1> tf_good_cb_good </h1>')
            elif j == 1:
                f.write('<h1> tf_good_cb_bad  </h1>')
            elif j == 2:
                f.write('<h1> tf_bad_cb_good </h1>')
            else:
                f.write('<h1> tf_bad_cb_bad  </h1>')

            for i in range(len(dict.keys())):
                #print("agwgwerghe", queries[j][i])
                f.write('<h2>' + str(queries[j][i]) + str(dict[(queries[j][i], passages[j][i])]) + '</h2>')
                f.write('<h3> kde </h3>')
                f.write(kde_heatmaps[i])
                f.write('<h3> absolute count </h3>')
                f.write(count_heatmaps[i])
                f.write('<h3> added values </h3>')
                f.write(sum_heatmaps[i])
            j += 1