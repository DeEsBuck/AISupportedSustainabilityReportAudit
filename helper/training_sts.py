import random
import logging as lg
import pandas as pd
from itertools import combinations
from sentence_transformers.cross_encoder import CrossEncoder

lg.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=lg.INFO)
log_name = lg.getLogger()

def create_train_data():
    # Excel laden
    # TODO ENUM Literal List
    df = pd.read_excel("resources/knowledgebase/datapoints/EFRAG_IG_3_List_of_ESRS.xlsx")
    
    # Zeilen mit fehlenden Angaben entfernen
    df = df.dropna(subset=["Name", "ESRS", "DR"])
    
    # Nur eindeutige Paragraphen
    df = df.drop_duplicates(subset=["Name"])
    
    # 🔁 Positive Paare: gleiche ESRS oder gleiche DR
    positive_pairs = []
    grouped = df.groupby("DR")  # oder "ESRS" als Alternative
    
    for _, group in grouped:
        names = group["Name"].tolist()
        for pair in combinations(names, 2):
            positive_pairs.append((pair[0], pair[1], 1.0))
    
    # 🔀 Negative Paare: zufällig aus verschiedenen DRs
    names = df["Name"].tolist()
    negative_pairs = []
    tries = 0
    while len(negative_pairs) < len(positive_pairs) and tries < 10 * len(positive_pairs):
        a, b = random.sample(names, 2)
        if df[df["Name"] == a]["DR"].values[0] != df[df["Name"] == b]["DR"].values[0]:
            negative_pairs.append((a, b, 0.0))
        tries += 1
    
    # 🔧 Kombinieren und mischen
    train_data = positive_pairs + negative_pairs
    random.shuffle(train_data)
    
    # Modell initialisieren
    model = CrossEncoder("distilroberta-base", num_labels=1)
    
    # Trainieren
    model.fit(train_data=train_data, epochs=1, batch_size=16)
    
    similarity = model.predict([("Describe the carbon footprint", "Report CO2 emissions in Scope 1 and 2")])
    print(f"Ähnlichkeitsscore: {similarity[0]:.2f}")


def save_trainings_model():
    # Excel laden
    df = pd.read_excel("resources/knowledgebase/datapoints/EFRAG_IG_3_List_of_ESRS.xlsx")
    df = df.dropna(subset=["Name", "DR"])  # oder "ESRS"
    df = df.drop_duplicates(subset=["Name"])
    
    # Positive Paare (gleiche DR)
    positive_pairs = []
    grouped = df.groupby("DR")
    
    for _, group in grouped:
        names = group["Name"].tolist()
        for pair in combinations(names, 2):
            positive_pairs.append({"sentence1": pair[0], "sentence2": pair[1], "label": 1.0})
    
    # Negative Paare (verschiedene DRs)
    names = df["Name"].tolist()
    negative_pairs = []
    tries = 0
    
    while len(negative_pairs) < len(positive_pairs) and tries < 10 * len(positive_pairs):
        a, b = random.sample(names, 2)
        dr_a = df[df["Name"] == a]["DR"].values[0]
        dr_b = df[df["Name"] == b]["DR"].values[0]
        if dr_a != dr_b:
            negative_pairs.append({"sentence1": a, "sentence2": b, "label": 0.0})
        tries += 1
    
    # Kombinieren und mischen
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    # Als DataFrame
    train_df = pd.DataFrame(all_pairs)
    
    # Export: Als Excel
    train_df.to_excel("resources/datapoints/prompts/esrs_training_data.xlsx", index=False)
    
    # Alternativ: CSV
    train_df.to_csv("resources/datapoints/prompts/esrs_training_data.csv", index=False)
    
    # Oder JSON
    train_df.to_json("resources/datapoints/prompts/esrs_training_data.json", orient="records", lines=True)


if __name__ == "__main__":
    create_train_data()
    save_trainings_model()
    