"""
Sample Dataset Generator
Creates a sample fake news dataset for demonstration purposes
"""

import pandas as pd
import numpy as np
import os
from config import DATA_DIR


def generate_sample_data():
    """
    Generate sample fake news dataset for demonstration
    
    Returns:
        pandas.DataFrame: Sample dataset
    """
    
    # Real news examples (label = 0)
    real_news = [
        "Scientists announce breakthrough in renewable energy technology at international conference.",
        "Government officials meet to discuss new infrastructure development plans for urban areas.",
        "Local community celebrates annual cultural festival with traditional performances and food.",
        "Research team discovers new species of marine life in deep ocean exploration mission.",
        "Education department announces scholarship programs for underprivileged students nationwide.",
        "Weather service issues advisory for heavy rainfall expected in coastal regions this week.",
        "Technology company launches new smartphone with advanced camera features and longer battery life.",
        "Health ministry initiates nationwide vaccination drive to combat seasonal flu outbreak.",
        "Sports federation announces schedule for upcoming national championship tournament next month.",
        "Central bank maintains interest rates unchanged in latest monetary policy review meeting.",
        "University researchers publish findings on climate change impacts in peer-reviewed journal.",
        "Transportation authority completes major highway expansion project ahead of schedule deadline.",
        "Agricultural department provides subsidies to farmers affected by recent drought conditions.",
        "International summit addresses global trade policies and economic cooperation between nations.",
        "Museum unveils new exhibition featuring historical artifacts from ancient civilizations discovered.",
        "Police department reports decrease in crime rates following community outreach initiatives.",
        "Stock market shows steady growth as investor confidence improves after economic reforms.",
        "Environmental agency launches program to protect endangered wildlife habitats in forests.",
        "Court delivers verdict in high-profile corruption case after lengthy trial proceedings.",
        "Hospital introduces advanced medical equipment to improve patient care and treatment outcomes.",
        "Space agency successfully launches satellite for weather monitoring and disaster management.",
        "City council approves budget for public park renovations and recreational facility upgrades.",
        "Trade organization reports increase in exports following new international business agreements.",
        "Scientists develop new vaccine candidate showing promising results in clinical trials.",
        "Government announces tax relief measures to support small and medium business enterprises.",
        "Aviation authority updates safety protocols following comprehensive industry standards review.",
        "Telecommunications company expands broadband internet access to rural and remote areas.",
        "Library system introduces digital lending platform for books and educational resources.",
        "Food safety agency conducts inspections to ensure quality standards in restaurants statewide.",
        "Energy department invests in solar power projects to reduce carbon emissions nationwide.",
    ]
    
    # Fake news examples (label = 1)
    fake_news = [
        "SHOCKING: Aliens spotted landing in major city center, government covering up evidence!!!",
        "BREAKING: Celebrity reveals secret to eternal youth, doctors hate this one trick!",
        "EXCLUSIVE: Billionaire caught on camera admitting world domination plans at secret meeting.",
        "URGENT: Drinking coffee causes immediate hair loss according to fake study nobody conducted.",
        "ALERT: New smartphone app can read minds and steal thoughts, delete now!!!",
        "VIRAL: Politician caught eating pizza with fork proves illuminati membership connection.",
        "LEAKED: Government planning to ban all social media platforms next week nationwide.",
        "EXPOSED: Vaccines contain microchips for population control according to internet rumor.",
        "BOMBSHELL: Scientists confirm earth is actually flat, all textbooks to be rewritten.",
        "WARNING: Eating bread daily causes 100% mortality rate eventually over many decades.",
        "SCANDAL: Famous actor secretly robot controlled by artificial intelligence technology.",
        "REVEALED: Drinking water dangerous, contains chemical compound known as hydrogen oxide.",
        "CONSPIRACY: Moon landing completely faked in Hollywood studio basement decades ago.",
        "PROOF: Evolution theory completely debunked by random blog post without evidence.",
        "UNBELIEVABLE: Lottery winner reveals winning numbers predicted by pet goldfish somehow.",
        "AMAZING: Man survives on sunlight alone for years, doctors baffled by this!",
        "CONFIDENTIAL: Government hiding cure for all diseases to profit from medical industry.",
        "DISTURBING: Popular restaurant chain using fake meat made from cardboard secretly.",
        "INCREDIBLE: Bermuda triangle mystery solved, aliens use it as parking space confirmed.",
        "SENSATIONAL: Time traveler from future warns about robot uprising happening next year.",
        "BREAKING: All birds are actually government surveillance drones watching everyone constantly.",
        "EXCLUSIVE: Tap water contains mind control chemicals making people believe fake news.",
        "SHOCKING: Antarctica hiding ancient alien civilization beneath ice discovered yesterday.",
        "WARNING: 5G towers causing immediate health problems, massive coverup by corporations ongoing.",
        "LEAKED: World leaders are shapeshifting reptilian aliens disguised as humans obviously.",
        "EXPOSED: Gravity doesn't exist, earth simply accelerating upward through space constantly.",
        "VIRAL: Eating organic food makes you magnetic and attracts metal objects instantly.",
        "BOMBSHELL: Sun actually cold, heat comes from air friction, science wrong entirely.",
        "ALERT: Chocolate cures all cancer types overnight, pharmaceutical companies hiding this.",
        "REVEALED: Dinosaurs never existed, all fossils planted by conspiracy theorists recently.",
    ]
    
    # Create DataFrame
    data = {
        'text': real_news + fake_news,
        'label': [0] * len(real_news) + [1] * len(fake_news)
    }
    
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def save_sample_data(filename='fake_news_sample.csv'):
    """
    Generate and save sample data to CSV
    
    Args:
        filename (str): Name of the output file
        
    Returns:
        str: Path to saved file
    """
    df = generate_sample_data()
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save to CSV
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    
    print(f"✓ Sample dataset saved to {filepath}")
    print(f"  Total samples: {len(df)}")
    print(f"  Real news: {sum(df['label'] == 0)}")
    print(f"  Fake news: {sum(df['label'] == 1)}")
    
    return filepath


def get_sample_texts_for_testing():
    """
    Get sample texts for testing the trained model
    
    Returns:
        list: List of test texts with their expected labels
    """
    test_samples = [
        {
            'text': "Scientists discover new treatment for diabetes in clinical trials.",
            'expected': 'Real',
            'category': 'Science'
        },
        {
            'text': "SHOCKING: Drinking water causes 100% death rate over lifetime!!!",
            'expected': 'Fake',
            'category': 'Sensational'
        },
        {
            'text': "Government announces new educational reforms for public schools.",
            'expected': 'Real',
            'category': 'Politics'
        },
        {
            'text': "EXPOSED: Celebrities are actually aliens in disguise, proof inside!",
            'expected': 'Fake',
            'category': 'Conspiracy'
        },
        {
            'text': "Local community raises funds for children's hospital renovation project.",
            'expected': 'Real',
            'category': 'Community'
        }
    ]
    
    return test_samples


if __name__ == "__main__":
    # Generate and save sample data
    save_sample_data()
    
    # Display sample texts
    print("\n" + "="*60)
    print("Sample Test Texts:")
    print("="*60)
    
    test_samples = get_sample_texts_for_testing()
    for i, sample in enumerate(test_samples, 1):
        print(f"\n{i}. {sample['text']}")
        print(f"   Expected: {sample['expected']} | Category: {sample['category']}")
