#!/usr/bin/python
# -*- coding: utf-8 -*-


from requests import get
from bs4 import BeautifulSoup
import re
import shutil
import os

base_url = "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/"

response = get(base_url)

soup = BeautifulSoup(response.text, 'html.parser')

matches = soup.find_all('a', attrs={"href": re.compile("tgz")})

if not os.path.exists('raw'): os.mkdir('raw')

def download_file(from_url, local_path):
    r = get(from_url, stream=True)
    with open(local_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)


for match in matches:
    file_url = os.path.join(base_url, match['href'])
    file_local = os.path.join('raw', match['href'])
    download_file(file_url, file_local)




from scipy.io import wavfile

wavfile.read('raw/1snoke-20120412-hge/wav/a0405.wav')

