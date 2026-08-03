import re 
import sys
import glob
import json
import pandas as pd
from pathlib import Path 




def get_test_batch(data: dict, error_type:str) -> list[dict]:
    """ Takes a list of tokens, pos, and error tags to be corrected and an error type
        and returns formatted prompts.

    Args:
        data (list[dict]): The data to be corrected  
        prompt_type (str): Corresponds to a prompt template  

    Returns:
        list[dict]: Our prompt batch
    """ 
    prompt_batch = []  
    tokens = data["token"]
    token_idx = data["token_ids"]
    pos = data["pos"]
    alt_pos = data["error"]
    error_ids = data["error_ids"]
        
    user_prompt = ""
    system_prompt = ""
        
    for idx in error_ids:
        # for each mismatch, get a formatted user prompt           
        system_prompt = get_system_prompt(tokens, pos, alt_pos[idx], idx, error_type)
        user_prompt = get_formatted_user_prompt(tokens, pos, alt_pos[idx], idx) 
            
        # append formatted prompt to prompt_batch
        message = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
        ]
        helper_dict = {
                "token_id": idx,
                "prompt_type":error_type,
                "message": message
        }
        prompt_batch.append(helper_dict)

    return prompt_batch





def write_to_file(records: list[dict], outfile: str):
    with open(outfile, "w") as out:
        for item in records:
            if "predictions" not in item:
                item["predictions"] = []
            out.write(json.dumps(item) + '\n')
    return




"""
Takes a pandas dataframe and extracts sentences, based on token ids.
Returns a list of dictionaries with format:
{'token_ids':[], 'token':[], 'pos':[], 'error':[]}

Input format (pandas dataframe):

GOLDPOS SID     TID     TOKEN   POS1    POS2
_       1       1       kurzer  ADJA    ADJA
_       1       2       /       $(      $(
_       1       3       doch    ADV     ADV
ADJA    1       4       gründlicher     ADJD    ADJA
_       1       5       Bericht NN      NN
...     ...     ...     ...     ...     ... 
"""
def df2json(df: pd.DataFrame) -> list[dict]:
    data = []
    sids = list(df.SID)
    tids = list(df.TID)
    toks = list(df.TOKEN)
    pos1 = list(df.POS1)
    pos2 = list(df.POS2)
    gold = list(df.GOLDPOS)
    dic = {'sid': -1, 'token_ids':[], 'token':[], 'pos':[], 'error':[], 'gold':[]}

    for i in range(len(sids)):
        if tids[i] == 1:
            if dic['pos'] != []:
                data.append(dic)
            # new sentence
            dic = {'sid': sids[i], 'token_ids':[], 'token':[], 'pos':[], 'error':[], 'gold':[]}
        dic['token_ids'].append(tids[i])
        dic['token'].append(toks[i])
        dic['pos'].append(pos1[i])
        dic['gold'].append(gold[i])
        
        if pos1[i] != pos2[i]:
            dic['error'].append(pos2[i])
        else:
            dic['error'].append("OK")
    
    # also get last sentence
    if dic['pos'] != []:
        data.append(dic)

    return data




def get_test_data(inpath: str) -> list[dict]:
    """ Function to extract our test data

    Args:
        data source

    Returns:
        data: list[dict] 

        [{'token_ids':[], 'token':[], 'pos':[], 'error':[]}]
    """
    # Open and read test data
    jsonl_data = []
    df = pd.read_csv(inpath, sep='\t', header=0)
    jsonl_data += df2json(df)
    print(f"Return jsonl data with {len(jsonl_data)} rows...")
    return jsonl_data



    
def get_system_prompt_baseline() -> str:
    return f"""
**Rolle und Ziel:**
Du bist Experte für Neuhochdeutsche und Frühneuhochdeutsche Sprache. Deine Aufgabe ist es, Wortarten in historischen Texten zu taggen.

**Kontext der Aufgabe:**
Du erhältst einen Textabschnitt aus einem neuhochdeutschen Text. Bestimme für jedes Wort die korrekte Wortart nach dem STTS-Tagset.

**Anweisungen:**
1.  Lies den bereitgestellten **Textabschnitt** sorgfältig durch. Der Text ist als Liste in einem JSON-Objekt formatiert.
3.  Analysiere alle Worte in ihrem grammatikalischen Kontext und weise jedem Wort sein korrektes POS-Tag zu. Die POS-Tags sollen als Liste "predictions" im JSON-Objekt eingefügt werden.
4.  Die Antwort soll nur das JSON-Objekt enthalten. Achte darauf, dass die Wortliste die gleiche Anzahl an Tokens hat wie die POS-Liste.
"""

    
def get_system_prompt(tokens: list, pos: list, pos2: str, idx: int, prompt_type:str) -> str: 
         
    if prompt_type == "GENERIC":
        return f"""
**Rolle und Ziel:**
Du bist Experte für Neuhochdeutsche und Frühneuhochdeutsche Sprache. Deine Aufgabe ist es, Fehler in der Zuweisung von POS-Tags zu korrigieren.

**Kontext der Aufgabe:**
Du erhältst einen Textabschnitt aus einem neuhochdeutschen Text. Im Textabschnitt ist ein Wort mit `<POS>`-Tags markiert. In der nächsten Zeile wird das Wort wiederholt, zusammen mit zwei möglichen POS-Tags. Entscheide, welches der beiden vorgeschlagenen Tags das richtige ist. Wenn beide Tags falsch sind, dann nenne das korrekte Tag aus dem STTS-Tagset.

**Anweisungen:**
1.  Lies den bereitgestellten **Textabschnitt** sorgfältig durch.
2.  Identifiziere das markierte **Wort**.
3.  Analysiere das Wort in seinem grammatikalischen Kontext. 
4.  Lies die beiden vorgeschlagenen POS-Tags und weise das korrekte Tag zu. Die Antwort soll nur das korrekte POS-Tag enthalten.
"""
    elif prompt_type == "NE_OTHER":
        return f"""
**Rolle und Ziel:**
Du bist Experte für Neuhochdeutsche und Frühneuhochdeutsche Sprache. Deine Aufgabe ist es, Fehler in der Zuweisung von POS-Tags zu korrigieren.

**Kontext der Aufgabe:**
Du erhältst einen Textabschnitt aus einem neuhochdeutschen Text. Im Textabschnitt ist ein Wort mit `<POS>`-Tags markiert. In der nächsten Zeile wird das Wort wiederholt, zusammen mit zwei möglichen POS-Tags. Entscheide, welches der beiden vorgeschlagenen Tags das richtige ist. Wenn beide Tags falsch sind, dann nenne das korrekte Tag aus dem STTS-Tagset.

**Anweisungen:**
1.  Lies den bereitgestellten **Textabschnitt** sorgfältig durch.
2.  Identifiziere das markierte **Wort**.
3.  Analysiere das Wort in seinem grammatikalischen Kontext. 
4.  Lies die beiden vorgeschlagenen POS-Tags und weise das korrekte Tag zu. Die Antwort soll nur das korrekte POS-Tag enthalten.
"""

    elif prompt_type == "FM_OTHER":
        return f"""
**Rolle und Ziel:**
Du bist Experte für Neuhochdeutsche und Frühneuhochdeutsche Sprache. Deine Aufgabe ist es, fremdsprachliches Material zu identifizieren.

**Aufgabe:**
Du erhältst den folgenden Input:
Text: eine kurze Textspanne aus einem frühneuhochdeutschen Text.
Wort: ein Wort in dieser Textspanne.

Entscheide, ob das Wort im Kontext deutsch oder fremdsprachlich (meist Latein) ist.
Antworte mit 'FM' für fremdsprachlich oder mit 'DE' für deutsch.
""" 

    elif prompt_type == "VVFIN_OTHER":
        return f"""
**Rolle und Ziel:**
Du bist Experte für Neuhochdeutsche und Frühneuhochdeutsche Sprache. Deine Aufgabe ist es, Fehler in der Zuweisung von POS-Tags zu korrigieren.

**Kontext der Aufgabe:**
Du erhältst einen Textabschnitt aus einem neuhochdeutschen Text. Im Textabschnitt ist ein Wort mit `<POS>`-Tags markiert. In der nächsten Zeile wird das Wort wiederholt, zusammen mit mehreren möglichen POS-Tags aus dem STTS-Tagset. Entscheide, welches der vorgeschlagenen Tags das richtige ist.

**Anweisungen:**
1.  Lies den bereitgestellten **Textabschnitt** sorgfältig durch.
2.  Identifiziere das markierte **Wort**.
3.  Analysiere das Wort in seinem grammatikalischen Kontext. 
4.  Lies die beiden vorgeschlagenen POS-Tags und weise das korrekte Tag zu. Die Antwort soll nur das korrekte POS-Tag enthalten.
""" 


    elif prompt_type == "PTKZU_OTHER":
        return f"""
**Rolle und Ziel:**
Du bist Experte für Neuhochdeutsche und Frühneuhochdeutsche Sprache. Deine Aufgabe ist es, Fehler in der Zuweisung von POS-Tags zu korrigieren.

**Kontext der Aufgabe:**
Du erhältst einen Textabschnitt aus einem neuhochdeutschen Text. Im Textabschnitt ist ein Wort mit `<POS>`-Tags markiert. In der nächsten Zeile wird das Wort wiederholt, zusammen mit mehreren möglichen POS-Tags aus dem STTS-Tagset. Entscheide, welches der vorgeschlagenen Tags das richtige ist.

**Anweisungen:**
1.  Lies den bereitgestellten **Textabschnitt** sorgfältig durch.
2.  Identifiziere das markierte **Wort**.
3.  Analysiere das Wort in seinem grammatikalischen Kontext. 
4.  Lies die beiden vorgeschlagenen POS-Tags und weise das korrekte Tag zu. Die Antwort soll nur das korrekte POS-Tag enthalten.
""" 


    elif prompt_type == "ADV_OTHER":
        return f"""
**Rolle und Ziel:**
Du bist Experte für Neuhochdeutsche und Frühneuhochdeutsche Sprache. Deine Aufgabe ist es, Fehler in der Zuweisung von POS-Tags zu korrigieren.

**Kontext der Aufgabe:**
Du erhältst einen Textabschnitt aus einem neuhochdeutschen Text. Im Textabschnitt ist ein Wort mit `<POS>`-Tags markiert. In der nächsten Zeile wird das Wort wiederholt, zusammen mit mehreren möglichen POS-Tags aus dem STTS-Tagset. Entscheide, welches der vorgeschlagenen Tags das richtige ist.

**Anweisungen:**
1.  Lies den bereitgestellten **Textabschnitt** sorgfältig durch.
2.  Identifiziere das markierte **Wort**.
3.  Analysiere das Wort in seinem grammatikalischen Kontext. 
4.  Lies die beiden vorgeschlagenen POS-Tags und weise das korrekte Tag zu. Die Antwort soll nur das korrekte POS-Tag enthalten.
""" 



def get_fewshot(prompt_type:str) -> str:

    if prompt_type == "NE_OTHER":
        return f"""
Beispiel 1: Der eine appolexia heißt und das andere <POS>Eppilencia<POS> .
Wortart zuweisen: ('Eppilencia', 'NE', 'NN')
Antwort: NE

Beispiel 2: Auqa stillatitia ex <POS>herba<POS> .
Wortart zuweisen: ('herba', 'NE', 'NN')
Antwort: FM

Beispiel 3: Der ägyptische Schlottendorn / in lateinischer Sprache <POS>Acacia<POS> genannt /
Wortart zuweisen: ('Acacia', 'NN', 'NE')
Antwort: NE

Beispiel 4: Dem wohlgeborenen Herrn / <POS>Doctor<POS> Paulo Khevenhüller von Aichelberg / Freiherr in LandtsCron /
Wortart zuweisen: ('Doctor', 'NE', 'NN')
Antwort: NN

Beispiel 5: Dem wohlgeborenen Herrn / Doctor Paulo Khevenhüller von Aichelberg / Freiherr in <POS>LandtsCron<POS> /
Wortart zuweisen: ('LandtsCron', 'NN', 'NE')
Antwort: NE

Beispiel 6: so wenig können auch ehrliche und erfahrene <POS>Medici<POS> solchen ihren Frevel approbieren
Wortart zuweisen: ('Medici', 'NE', 'NN')
Antwort: NN
"""

    # no fewshot examples for this error type
    elif prompt_type == "FM_OTHER":
        return f""" 
"""


    elif prompt_type == "VVFIN_OTHER":
        return f"""
Beispiel 1: tu <POS>es<POS> auf die Feigwarzen.
Wortart zuweisen: ('es', 'VVFIN', 'PPER')
Antwort: PPER

Beispiel 1: Ist gut <POS>gebraucht<POS> dem entzündeten Podagra
Wortart zuweisen: ('gebraucht', 'VVFIN', 'VVPP')
Antwort: VVPP

Beispiel 1: Und wird <POS>vermischt<POS> den Arzneien
Wortart zuweisen: ('vermischt', 'VVPP', 'VVFIN')
Antwort: VVPP
"""


    elif prompt_type == "PTKZU_OTHER":
        return f"""
Beispiel 1: ist eine goldene Arznei <POS>zu<POS> unzähligen Krankheiten /
Wortart zuweisen: ('zu', 'PKTZU', 'APPR')
Antwort: PPER

Beispiel 2: / unten aber auf der Seite gegen der Erden <POS>zu<POS> ganz weiß
Wortart zuweisen: ('zu', 'PTKZU', 'APPR')
Antwort: APZR

Beispiel 3: Also hat es auch seine Bedeutung / Nutzen und Brauch <POS>zu<POS> der Seelen Arznei .
Wortart zuweisen: ('zu', 'PTKZU', 'APPR')
Antwort: APPR
"""

    elif prompt_type == "ADV_OTHER":
        return f"""
Beispiel 1: als have ich / <POS>kürzlich<POS> hiervon meine Meinung niemands vorgegriffen /
Wortart zuweisen: ('kürzlich', 'ADV', 'ADJD')
Antwort: ADV

Beispiel 2: oder einem Schattenhut <POS>gleich<POS> /
Wortart zuweisen: ('gleich', 'ADV', 'ADJD')
Antwort: ADJD

Beispiel 3: und Saft von der Zeitlosen Wurzel jegliches <POS>gleich<POS> viel
Wortart zuweisen: ('vermischt', 'ADJD', 'ADV')
Antwort: ADJD

Beispiel 4: und <POS>erstlich<POS> folgt das Kräutlein Abbiss /
Wortart zuweisen: ('erstlich', 'ADV', 'ADJD')
Antwort: ADV

Beispiel 5: danach mische <POS>ferner<POS> guten Zucker dazu /
Wortart zuweisen: ('ferner', 'ADV', 'ADJD')
Antwort: ADV
"""




def get_formatted_user_prompt(tokens:list, pos: list, alt_pos: str, idx: int) -> str:

    token_str = " ".join(["<POS>" + tokens[i] + "<POS>" if i == idx else tokens[i] for i in range(len(tokens))])
    return f"""
Textabschnitt: {token_str}

Wortart zuweisen: {tokens[idx], pos[idx], alt_pos}

Antwort: 
"""



def get_formatted_user_prompt_baseline(tokens:dict) -> str:

    return f"""
Textabschnitt: {tokens}  
"""




def get_baseline_batch(data: dict, prompt_type:str) -> list[dict]:
    """ Returns a list of tokens for POS prediction and goldpos for evaluation as formatted prompts.

    Args:
        data (list[dict]): The data to be tagged  
        prompt_type (str): specify that this is the baseline

    Returns:
        list[dict]: Our prompt batch
    
    
    """
    prompt_batch = []  
    tokens = data["token"]
    token_idx = data["token_ids"]
    pos = data["pos"]
    alt_pos = data["error"] 
        
    user_prompt = ""
    system_prompt = ""

    # for each item, get a formatted user prompt                 
    system_prompt = get_system_prompt_baseline()
    user_prompt = get_formatted_user_prompt_baseline({'Text': tokens}) 
            
    # append formatted prompt to prompt_batch
    message = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
    ]
    helper_dict = { 
                "prompt_type":prompt_type,
                "message": message
    }
    prompt_batch.append(helper_dict)

    return prompt_batch

     


