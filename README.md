# Manga Text Bubble Detector to Translate
This is a repository for a deep learning model that can detect text bubbles in manga and feed them into a translator.


### Work Flow
1. single manga page =>
2. text bubble detector to crop text bubbles (UNet trained with personal dataset and manga101) =>
3. optical character recognition engine to extract text from bubbles(tesseract OCR) => 
4. google translate API

The text bubble detector is effective while the OCR is not effective (especially for vertical Japanese). Therefore, the translation result is not good.

![alt text]([wd_lab/v4_13600/#cont/003l_cont.jpg](https://github.com/VincentQQu/manga_text_bubble_detect_translate/blob/main/wd_lab/v4_13600/%23cont/003l_cont.jpg))
![alt text]([wd_lab/v4_13600/#cont/003r_cont.jpg](https://github.com/VincentQQu/manga_text_bubble_detect_translate/blob/main/wd_lab/v4_13600/%23cont/003l_cont.jpg))

### How to Use
You can see some example outputs in exds/v4_13600/
1. clear the folder exds/v4_13600/
2. put the manga pages (better single pages in resolution 250x750) you want to translate into the folder.
3. python3 word_detect.py to generate text bubbles
4. python3 word_detect_and_translate.py to generate text bubbles and translate
