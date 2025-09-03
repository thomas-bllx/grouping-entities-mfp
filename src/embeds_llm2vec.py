import gc
import os
import pickle
from glob import glob

import pandas as pd
import torch
from llm2vec import LLM2Vec
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoTokenizer

torch.cuda.empty_cache()
torch.cuda.empty_cache()

gc.collect()
gc.collect()
gc.collect()


def prepare_model(MODEL_ID):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        # cache_dir="hf_model_cache/",
    )

    model = PeftModel.from_pretrained(
        model,
        MODEL_ID,
    )

    llm2vec_model = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=128)

    return llm2vec_model


# MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"

MODEL_ID = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
batch_size = 16

llm2vec_model = prepare_model(MODEL_ID=MODEL_ID)


def get_ontology_facet_colon(ontology_file):
    df = pd.read_csv(ontology_file, sep="\t")
    facet_colon_property_list = df["facet_property"].str.strip().unique()

    return facet_colon_property_list


def get_wikidata_facet_colon(wikidata_file):
    df = pd.read_csv(wikidata_file, sep="\t")
    facet_colon_property_list = df["facet_property"].str.strip().unique()

    return facet_colon_property_list


wikidata_facet_prop = "data/wikidata_facet_prop/*_parsed_all_cols.tsv"
facet_colon_property_files = glob(wikidata_facet_prop)


print(f"wikidata_files_all")
print(f"{facet_colon_property_files}")


for fact_property_file in facet_colon_property_files:
    print(f"getting_embeddings: {fact_property_file}", flush=True)

    # with open(fact_property_file, "r") as fin:
    #     facet_property = [fp.strip("\n") for fp in fin.readlines()]

    # for ontology data
    # facet_property = get_ontology_facet_colon(fact_property_file)

    # for wiki_data files
    facet_property = get_wikidata_facet_colon(fact_property_file)

    print(f"num_facet_property: {len(facet_property)}")

    facet_property = [str(item) for item in facet_property]

    facet_property_embeds = (
        llm2vec_model.encode(facet_property, batch_size=batch_size)
        .detach()
        .cpu()
        .numpy()
    )

    print(f"facet_property_embeds.shape: {facet_property_embeds.shape}")

    facet_property_and_embedding = [
        (facet_prop, embed)
        for facet_prop, embed in zip(facet_property, facet_property_embeds)
    ]

    facet_property_and_embedding_dict = {
        facet_prop: embed
        for facet_prop, embed in zip(facet_property, facet_property_embeds)
    }

    print(f"Top 5 facet_property_and_embedding")
    print(facet_property_and_embedding[0:5])

    # output_dir_path = os.path.dirname(fact_property_file)
    file_name_with_ext = os.path.basename(fact_property_file)
    file_name, file_extension = os.path.splitext(file_name_with_ext)

    out_file_name = os.path.join(
        "wikidata", os.path.basename(MODEL_ID), f"{file_name.replace("parsed_all_cols", "facet_colon_property")}_embeds.pkl"
    )
    pickle_output_file = os.path.join("embeds", out_file_name)

    with open(pickle_output_file, "wb") as pkl_file:
        pickle.dump(facet_property_and_embedding_dict, pkl_file)

    print(f"got_embeddings: {fact_property_file}", flush=True)
    print(f"embeds_saved: {pickle_output_file}", flush=True)
    print(flush=True)

torch.cuda.empty_cache()
torch.cuda.empty_cache()

gc.collect()
gc.collect()
gc.collect()