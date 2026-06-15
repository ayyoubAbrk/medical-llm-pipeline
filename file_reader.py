from pypdf import PdfReader
import os

reader = PdfReader("cr.pdf")
text = ""

for page in reader.pages:
    text_page = page.extract_text()
    if text_page:
        text += text_page
    

print(text)
