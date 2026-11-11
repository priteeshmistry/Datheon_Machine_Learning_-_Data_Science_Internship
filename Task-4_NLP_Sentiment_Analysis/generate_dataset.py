#!/usr/bin/env python3
"""
Generate a diverse, realistic 1,200-review dataset for Task 4 Sentiment Analysis,
combining the user's 10 starter reviews with rich consumer product, software,
and entertainment reviews.
"""

import os
import random
import pandas as pd

random.seed(42)

TASK_DIR = r"D:\Datheon Internship\Task-4_NLP_Sentiment_Analysis"
STARTER_CSV = os.path.join(TASK_DIR, "starter_reviews.csv")
CORPUS_CSV = os.path.join(TASK_DIR, "customer_reviews_corpus.csv")

# Load existing starter reviews
starter_df = pd.read_csv(STARTER_CSV)
print(f"[*] Loaded {len(starter_df)} starter reviews from {STARTER_CSV}")

# Templates and vocabulary components for synthetic generation
positive_templates = [
    "The {product} is {pos_adj}! {feature} works {pos_adv} and exceeded all my expectations.",
    "Absolutely love this {product}. It has {pos_adj} build quality and {feature} is seamless.",
    "Great {product} for the price. Fast delivery, {pos_adj} customer support, and very reliable.",
    "I was hesitant at first, but this {product} turned out to be {pos_adj}. Highly recommend to everyone!",
    "Super clean interface and {pos_adj} performance. The {feature} makes daily workflow effortless.",
    "Best purchase I have made all year! Top-tier {feature} and the quality feels extremely durable.",
    "Five stars! The {product} arrived earlier than promised and was packaged with great care.",
    "Outstanding performance! {feature} is super fast and battery life lasts all day without issues.",
    "Very impressed with the responsiveness of customer service. Resolved my setup issue in minutes.",
    "The build is solid, the {feature} is intuitive, and it looks beautiful on my desk.",
    "Brilliant design and {pos_adj} execution. Everything works straight out of the box with zero fuss.",
    "Incredible value for money! I have recommended this {product} to all my colleagues.",
    "A game-changer for my daily productivity. {feature} saves me hours every single week.",
    "Exceeded expectations in every regard. High quality materials, {pos_adj} sound, and sleek finish.",
    "Delivered promptly and functioned flawlessly from day one. Truly a fantastic product!"
]

negative_templates = [
    "Terrible {product}! Stopped working after {timeframe} and customer support was completely useless.",
    "Very disappointed with this purchase. The {product} feels {neg_adj} and {feature} constantly fails.",
    "Do not waste your money on this {product}. Plenty of {pos_adj} alternatives available for cheaper.",
    "Arrived {neg_adj} with missing parts. Return process was frustrating and slow.",
    "The {feature} is completely broken and lags terribly. Extremely frustrated with this item.",
    "Cheap plastic construction and poor quality control. Broke during the very first week.",
    "Misleading product description! The {product} lacks the advertised {feature} and battery drains fast.",
    "Customer service was rude and refused to issue a refund for the defective {product}.",
    "Worst experience ever. Delivery was delayed by two weeks and package arrived crushed.",
    "Overheating issues and constant software crashes. Definitely returning this tomorrow.",
    "Unusable software and complicated setup instructions. Totally regret buying this.",
    "Poor performance, terrible build quality, and zero customer support. Avoid at all costs!",
    "The {product} stopped charging after three days. Complete waste of hard-earned money.",
    "Flimsy materials, noisy operation, and the {feature} barely functions. Very unhappy.",
    "Subpar quality that does not justify the high price tag. Very dissatisfied."
]

products = [
    "laptop", "smartphone", "wireless earbuds", "mechanical keyboard", "monitor",
    "smartwatch", "coffee maker", "noise-canceling headphones", "external SSD",
    "gaming mouse", "bluetooth speaker", "tablet", "office chair", "webcam",
    "power bank", "smart thermostat", "fitness tracker", "robot vacuum", "microphone"
]

features = [
    "battery life", "sound quality", "display brightness", "connectivity",
    "setup process", "touchscreen response", "ergonomic design", "software ecosystem",
    "charging speed", "noise isolation", "data transfer rate", "build craftsmanship"
]

pos_adjs = ["exceptional", "phenomenal", "superb", "brilliant", "fantastic", "flawless", "durable", "premium", "top-notch", "delightful", "wonderful", "terrific", "impressive", "reliable"]
neg_adjs = ["horrible", "damaged", "cheap", "defective", "flimsy", "unreliable", "terrible", "awful", "frustrating", "broken", "subpar", "dreadful"]
pos_advs = ["flawlessly", "smoothly", "brilliantly", "reliably", "effortlessly", "wonderfully", "seamlessly"]
timeframes = ["two days", "one week", "a few hours", "three days", "ten days", "two weeks"]

# HTML tags and noisy artifacts to simulate real-world web scrape text
noise_tags = ["", "<br />", "<br /><br />", "<b>", "</b>", "<p>", "</p>", "...", "!!", " :)", " :("]

reviews = []

# First, add the starter reviews
for _, row in starter_df.iterrows():
    # Only keep positive and negative for binary classification, or keep as is
    reviews.append({
        "review_id": int(row["review_id"]),
        "review_text": str(row["review_text"]),
        "sentiment": str(row["sentiment"]).lower()
    })

current_id = len(reviews) + 1

# Generate 600 positive and 600 negative reviews
for _ in range(600):
    tmpl = random.choice(positive_templates)
    txt = tmpl.format(
        product=random.choice(products),
        feature=random.choice(features),
        pos_adj=random.choice(pos_adjs),
        pos_adv=random.choice(pos_advs)
    )
    # Add random noise
    if random.random() < 0.25:
        txt = random.choice(noise_tags) + " " + txt + " " + random.choice(noise_tags)
    reviews.append({
        "review_id": current_id,
        "review_text": txt.strip(),
        "sentiment": "positive"
    })
    current_id += 1

for _ in range(600):
    tmpl = random.choice(negative_templates)
    txt = tmpl.format(
        product=random.choice(products),
        feature=random.choice(features),
        pos_adj=random.choice(pos_adjs),
        neg_adj=random.choice(neg_adjs),
        timeframe=random.choice(timeframes)
    )
    if random.random() < 0.25:
        txt = random.choice(noise_tags) + " " + txt + " " + random.choice(noise_tags)
    reviews.append({
        "review_id": current_id,
        "review_text": txt.strip(),
        "sentiment": "negative"
    })
    current_id += 1

# Add 40 realistic neutral reviews as well
neutral_templates = [
    "The {product} is okay for basic tasks. Nothing special, average performance.",
    "Decent build, but the {feature} could be improved. Fair for the discounted price.",
    "Received as expected. It does the job adequately, neither impressive nor terrible.",
    "Average quality {product}. Meets minimum requirements but lacks advanced features.",
    "Standard shipping and ordinary packaging. It works as described, no major complaints."
]
for _ in range(40):
    tmpl = random.choice(neutral_templates)
    txt = tmpl.format(
        product=random.choice(products),
        feature=random.choice(features)
    )
    reviews.append({
        "review_id": current_id,
        "review_text": txt.strip(),
        "sentiment": "neutral"
    })
    current_id += 1

# Shuffle records (keeping reproducible seed)
random.shuffle(reviews)
# Re-index review_id sequentially
for idx, r in enumerate(reviews, start=1):
    r["review_id"] = idx

corpus_df = pd.DataFrame(reviews)
corpus_df.to_csv(CORPUS_CSV, index=False)
print(f"[SUCCESS] Generated {len(corpus_df)} reviews saved to: {CORPUS_CSV}")
print("Class breakdown:")
print(corpus_df["sentiment"].value_counts())
