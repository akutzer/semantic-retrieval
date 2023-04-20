import os
import subprocess
import urllib.request
import shutil
from py7zr import unpack_7zarchive
import pandas as pd



# enables shutil to unpack 7zip files
shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
REDOWNLOAD = True


if __name__ == "__main__":

    # https://<wiki-name>.fandom.com/wiki/Special:Statistics
    WIKI_DUMPS_URLS = {
        "harry_potter": "https://s3.amazonaws.com/wikia_xml_dumps/h/ha/harrypotter_pages_current.xml.7z",
        "elder_scrolls": "https://s3.amazonaws.com/wikia_xml_dumps/e/el/elderscrolls_pages_current.xml.7z"
    }


    # create new directories for the data dumps in the data/ directory
    preprocessing_path = "../../data/preprocessing/"
    dump_path = os.path.join(preprocessing_path, "dumps")
    extraction_path = os.path.join(dump_path, "tmp")
    os.makedirs(extraction_path, exist_ok=True)


    for wiki_name, dump_url in WIKI_DUMPS_URLS.items():
        # download the data dump into the data/preprocessing/dumps directory
        dump_file_name = dump_url.split("/")[-1]
        path_to_dump_archive = os.path.join(dump_path, dump_file_name)
        path_to_dump_file = os.path.splitext(path_to_dump_archive)[0]

        # download and extract the dump, if it doesn't exist or a re-download is requested
        if not os.path.isfile(path_to_dump_archive) or REDOWNLOAD:
            urllib.request.urlretrieve(dump_url, filename=path_to_dump_archive, )

            # convert relative to absolute paths, because py7zr doesn't work otherwise
            abs_path_to_dump_archive = os.path.abspath(path_to_dump_archive)
            abs_dump_path = os.path.abspath(dump_path)
            # unpack the data dump
            shutil.unpack_archive(filename=abs_path_to_dump_archive, extract_dir=abs_dump_path)

        # use wikiextractor to extract and clean the data dumps
        # this created many json like files in the extraction_path/AA directory
        cmd_str = f"python3 -m wikiextractor.WikiExtractor --json -o {extraction_path} {path_to_dump_file}"
        subprocess.run(cmd_str, shell=True)

        # create dataframe and add the content of all data files to it
        df = pd.DataFrame()
        for root, dirs, files in os.walk(extraction_path):
            for f in files:
                path_f = os.path.join(root, f)
                df = pd.concat([df, pd.read_json(path_f, lines=True)], ignore_index=True, sort=False)
        print(f"{df.shape[0]} pages found!")

        # drop wiki pages which are to short
        df = df[df.text.map(len) >= 50]
        # print(df.text.map(len).describe())

        # save the dataframe to a json file
        print(f"{df.shape[0]} pages remain after cleaning!")
        wiki_json_file = os.path.join(preprocessing_path, wiki_name) + ".json"
        df.to_json(path_or_buf=wiki_json_file, orient="records", indent=1)

        # delete the tmp folder filled by temporary files from the wikiextractor
        shutil.rmtree(extraction_path)
        os.makedirs(extraction_path, exist_ok=True)


    # finally remove the tmp folder
    shutil.rmtree(extraction_path)
