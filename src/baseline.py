import pandas as pd
import re

PAT_SENTIMENT_GOOD = r"amazing|excellent|wonderful|fantastic|perfect|great|lovely|awesome|beautiful|superb|outstanding|fabulous|incredible|brilliant|exceptional|enjoyed|loved|pleasant|delightful|recommend|happy|satisfied|would stay again|had a great time|loved our stay|exceeded expectations|everything was perfect"
PAT_SENTIMENT_BAD = r"terrible|awful|horrible|bad|poor|worst|disappointing|disappointed|unpleasant|unacceptable|frustrating|uncomfortable|dirty|regret|problem|issues|complaint|avoid|rude|would not recommend|never again|not worth it|wouldn't stay again|did not enjoy"
PAT_SENTIMENT_NEUTRAL = r"okay|ok|average|fine|fair|acceptable|normal|ordinary|nothing special|so-so|could be better|as expected"

PAT_ACCESS_GOOD = r"central|city center|city centre|well located|convenient|accessible|walkable|walking distance|close to|near|nearby|easy access|public transport|metro nearby|heart of|short walk|minutes walk|easy to reach|well connected"
PAT_ACCESS_BAD = r"far from|far away|remote|isolated|inconvenient|difficult to reach|hard to reach|long walk|too far|not central|not accessible|difficult access|far from metro|far from center"
PAT_ACCESS_NEUTRAL = r"location was okay|location was fine|reasonable location|location was acceptable|not too far|not too close|moderate distance|location was alright"

PAT_QUIET = r"quiet|very quiet|peaceful|silent|tranquil|calm|soundproof|no noise|no disturbance|not noisy|quiet neighborhood|peaceful nights|no street noise|well insulated|couldn't hear anything"
PAT_NOISY = r"noisy|very noisy|too noisy|loud|very loud|traffic noise|street noise|road noise|construction noise|loud music|disturbed sleep|noise at night|loud neighbors|noise problem|unbearable noise"
PAT_NOISE_NEUTRAL = r"some noise|a little noisy|slightly noisy|bit noisy|minor noise|not too noisy|manageable noise|tolerable noise|noise was okay|noise was fine|noise was acceptable|didn't mind the noise"

EVAL_DICT = {
    "general_sentiment": eval_sentiment,
    "accessibility": eval_accessibility,
    "noise": eval_noise
}

def eval_sentiment(rv):
    rv = rv.copy()
    sentiment_good = rv["comments"].str.findall(PAT_SENTIMENT_GOOD, flags=re.I)
    sentiment_bad = rv["comments"].str.findall(PAT_SENTIMENT_BAD, flags=re.I)
    # sentiment_neutral = rv["comments"].str.findall(PAT_SENTIMENT_NEUTRAL, flags=re.I)

    nums_of_good_sentiment = sentiment_good.map(lambda x: len(x))
    nums_of_bad_sentiment = sentiment_bad.map(lambda x: len(x))
    # nums_of_neutral_sentiment = sentiment_neutral.map(lambda x: len(x))
    sentiment_score = nums_of_good_sentiment - nums_of_bad_sentiment

    rv["sentiment"] = ""
    rv.loc[sentiment_score > 0, "sentiment"] = "Positive"
    rv.loc[sentiment_score < 0, "sentiment"] = "Negitive"
    rv.loc[sentiment_score == 0, "sentiment"] = "Mixed" 
    return rv

def eval_accessibility(rv):
    rv = rv.copy()
    access_good = rv["comments"].str.contains(PAT_ACCESS_GOOD, flags=re.I)
    access_bad = rv["comments"].str.contains(PAT_ACCESS_BAD, flags=re.I)
    access_neutral = rv["comments"].str.contains(PAT_ACCESS_NEUTRAL, flags=re.I)

    rv["accessibility"] = "Not mentioned"
    rv.loc[access_good, "accessibility"] = "Highly accessible"
    rv.loc[access_bad, "accessibility"] = "Poor access"
    rv.loc[access_neutral, "accessibility"] = "Neutral"
    return rv

def eval_noise(rv):
    rv = rv.copy()
    noise_quiet = rv["comments"].str.contains(PAT_QUIET, flags=re.I)
    noise_noisy = rv["comments"].str.contains(PAT_NOISY, flags=re.I)
    noise_neutral = rv["comments"].str.contains(PAT_NOISE_NEUTRAL, flags=re.I)

    rv["noise"] = "Not mentioned"
    rv.loc[noise_quiet, "noise"] = "Quiet"
    rv.loc[noise_noisy, "noise"] = "Noisy"
    rv.loc[noise_neutral, "noise"] = "Acceptable"
    return rv

def get_aspect(config):
    try:
        ASPECTS = config.get("llm").get("aspects")
    except Exception:
        raise Exception("Có lỗi khi lấy 'aspects' ở mục 'llm' từ file config")

def baseline(rv_to_label, config_path): 
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    rv = rv_to_label.copy()
    ASPECTS = get_aspect(config)

    for aspect in ASPECTS:
        if aspect not in ASPECTS.keys():
            raise Exception(f"Chưa có hàm đánh giá aspect {aspect}")
        else:
            rv = EVAL_DICT[aspect](rv)

    return rv