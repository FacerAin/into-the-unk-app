import streamlit as st
import pandas as pd
from transformers import AutoTokenizer

###################################
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
from st_aggrid import GridUpdateMode, DataReturnMode

###################################

model_list = [
    "klue/roberta-base",
    "monologg/koelectra-base-v3-discriminator",
    "beomi/KcELECTRA-base",
]


def extract_sentences(df, cols):
    sentences = []
    for col in cols:
        for item in df[col]:
            sentences.append(item)
    return sentences


def tokenize_sentences(sentences, tokenizer):
    original_sentences = []
    tokenized_sentences = []
    for sentence in sentences:
        result = tokenizer.tokenize(sentence)

        if "[UNK]" in result:
            original_sentences.append(sentence)
            tokenized_sentences.append(result)
    return original_sentences, tokenized_sentences


st.set_page_config(page_icon="❄️", page_title="CSV Wrangler", layout="wide")

# st.image("https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/balloon_1f388.png", width=100)
st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/320/apple/325/snowflake_2744-fe0f.png",
    width=100,
)

st.title("Into the Unknown")


c29, c30, c31 = st.columns([1, 6, 1])

with c30:

    uploaded_file = st.file_uploader(
        "Upload Your CSV FILE",
        key="1",
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        cols = df.columns
        options = st.multiselect(
            "분석할 컬럼들을 선택해주세요.",
            cols,
        )
        model_name = st.selectbox(
            "사용할 토크나이저를 선택해주세요.",
            model_list,
        )
        special_token_text = st.text_area(",(콤마) 단위로 Vocab에 추가할 토큰을 입력해주세요.")
        sentences = extract_sentences(df, options)
        run = st.button("분석!")

    else:
        st.info(
            f"""
                👆 Upload a .csv file first.
                """
        )

        st.stop()


if run:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(special_token_text.split(","))

    original_sentences, tokenized_sentences = tokenize_sentences(sentences, tokenizer)
    print(tokenized_sentences)
    st.info(
        f"""
            👆 총 {len(tokenized_sentences)}개의 UNK 토큰을 포함한 문장이 있습니다
            """
    )
    data_df = pd.DataFrame(
        {"sentence": original_sentences, "tokenized": tokenized_sentences}
    )
    st.dataframe(data_df)
