import json
import os

paths_input = ["../../data/fandom_qa_2/harry_potter_qa/harry_potter.json",
                "../../data/fandom_qa_2/elder_scrolls_qa/elder_scrolls.json",
                "../../data/fandom_qa_2/witcher_qa/witcher.json"
               ]
paths_output = ["../../data/fandom_qa_2/harry_potter_qa",
                "../../data/fandom_qa_2/elder_scrolls_qa",
                "../../data/fandom_qa_2/witcher_qa"
                ]
wiki_names = ["harry", "elder", "witcher"]
wiki_page_min_paragraph_count = 6
paragraph_min_character_count = 600
distribution = [600, 400, 200]
indexes_for_distribution = [600, 1000, 1200]
count_people = 5

if __name__ == "__main__":
    pid = 0
    for i in range(0, len(paths_input)):
        # Open the JSON file and load its contents into a variable
        with open(paths_input[i], 'r') as f:
            data = json.load(f)

        # Create an empty dictionary to store the output
        output = {}

        # Loop through each dictionary in the list

        p_index = 0
        person_id = 0
        for item in data:
            # Extract the relevant keys and values
            #id = item['id']
            #revid = item['revid']
            url = item['url']
            title = item['title']
            text = item['text']
            if len(text) < wiki_page_min_paragraph_count:
                continue

            # Add the key-value pair to the output dictionary
            for p in text:
                if len(p) < paragraph_min_character_count:
                    continue
                output[pid] = {
                    'title': title,
                    'url': url,
                    'passage': p,
                    'pos_query': ["...", "...", "..."],
                    'neg_query': ["...", "...", "..."]
                }
                pid += 1
                p_index += 1
                if p_index == distribution[i]:
                    person_id += 1
                    p_index = 0
                    # Print the output dictionary
                    print("person", person_id)
                    print("path:", paths_input[i])
                    print("how many paragraphs:", len(output))

                    # Write the output dictionary to a JSON file
                    filename = wiki_names[i] + "_pqq_blueprint" + str(person_id) + ".json"
                    with open(os.path.join(paths_output[i], filename), mode="w", encoding="utf-8") as f:
                        json.dump(output, f, indent=0)
                    output = {}
            if person_id >= count_people:
                break
