#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of ToMBench parser functions
"""

import json
import os
import math
import re
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parents[1] / "data"

# Define the standard cognitive dimensions
STANDARD_DIMENSIONS = {
    'emotion': 'Emotion',
    'desire': 'Desire', 
    'intention': 'Intention',
    'knowledge': 'Knowledge',
    'belief': 'Belief',
    'non-literal communication': 'Non-Literal Communication'
}

def clean_option_content(option_content, option_letter):
    """
    Clean option content by removing various prefixes and extra whitespace
    """
    if not option_content:
        return ""
    
    # Convert to string if needed
    option_content = str(option_content)
    
    # Remove various prefix patterns
    prefixes_to_remove = [
        f"{option_letter}. ",      # "A. "
        f"{option_letter}.",       # "A."
        f"{option_letter}: ",      # "A: "
        f"{option_letter}:",       # "A:"
        f" {option_letter}. ",    # " A. " (with leading space)
        f" {option_letter}.",     # " A." (with leading space)
        f" {option_letter}: ",    # " A: " (with leading space)
        f" {option_letter}:",     # " A:" (with leading space)
    ]
    
    # Try to remove each prefix pattern
    for prefix in prefixes_to_remove:
        if option_content.startswith(prefix):
            option_content = option_content[len(prefix):]
            break
    
    # Clean up extra whitespace
    option_content = option_content.strip()
    
    return option_content

def parse_tombench_jsonl(file_path):
    """
    Read and parse a ToMBench JSONL file, extracting specific fields:
    - ability: The cognitive ability being tested
    - story: The story context (Chinese)
    - index: The question index
    - question: The question (Chinese)
    - options: A, B, C, D options (Chinese)
    - answer: The correct answer
    """
    cn_entries = []
    en_entries = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON line
                entry = json.loads(line.strip())
                
                # Extract and parse ability field into dimension and ability
                raw_ability = entry.get('能力\nABILITY', '')
                abilities_list = []
                
                if raw_ability:
                    # Find all dimension positions first
                    dimension_positions = []
                    for key, standard in STANDARD_DIMENSIONS.items():
                        # Find all occurrences of this dimension
                        start = 0
                        while True:
                            pos = raw_ability.lower().find(key, start)
                            if pos == -1:
                                break
                            # Make sure this is actually a dimension (followed by colon)
                            colon_pos = raw_ability.find(':', pos)
                            if colon_pos != -1 and colon_pos < pos + len(key) + 5:  # Within reasonable distance
                                dimension_positions.append((pos, key, standard))
                            start = pos + 1
                    
                    # Sort by position
                    dimension_positions.sort(key=lambda x: x[0])
                    
                    # Extract abilities for each dimension
                    for i, (pos, key, standard) in enumerate(dimension_positions):
                        # Find the colon after this dimension
                        colon_pos = raw_ability.find(':', pos)
                        if colon_pos != -1:
                            # Find where this ability ends (next dimension or end of string)
                            next_pos = len(raw_ability)
                            if i + 1 < len(dimension_positions):
                                # Look for the next dimension that actually starts a new ability
                                next_dim_pos = dimension_positions[i + 1][0]
                                # Make sure there's a colon after the next dimension
                                next_colon_pos = raw_ability.find(':', next_dim_pos)
                                if next_colon_pos != -1:
                                    next_pos = next_dim_pos
                            
                            # Extract ability text between colon and next dimension
                            ability_text = raw_ability[colon_pos + 1:next_pos].strip()
                            
                            if ability_text:
                                abilities_list.append({
                                    'dimension': standard,
                                    'ability': ability_text
                                })
                    
                    # If no patterns found, try simple colon split as fallback
                    if not abilities_list and ':' in raw_ability:
                        dimension, specific_ability = raw_ability.split(':', 1)
                        abilities_list.append({
                            'dimension': dimension.strip(),
                            'ability': specific_ability.strip()
                        })
                    
                    # If still no patterns found, treat as general ability
                    if not abilities_list:
                        abilities_list.append({
                            'dimension': 'Unknown',
                            'ability': raw_ability
                        })
                
                # Extract Chinese fields
                chinese_entry = {
                    # 'id': len(cn_entries) + 1,
                    'abilities': abilities_list,  # Now it's a structured dict
                    'story': entry.get('故事', ''),
                    'index': entry.get('序号\nINDEX', ''),
                    'question': entry.get('问题', ''),
                    'options': {},
                    'answer': entry.get('答案\nANSWER', ''),
                    # 'sheet_name': entry.get('sheet_name', ''),
                    'language': 'Chinese'
                }
                
                # Extract Chinese options
                for key in ['选项A', '选项B', '选项C', '选项D']:
                    if key in entry and entry[key]:
                        # Check if the value is NaN (without importing math)
                        if isinstance(entry[key], float) and str(entry[key]) == 'nan':
                            continue  # Skip NaN values
                        option_letter = key[-1]  # Extract A, B, C, or D
                        option_content = entry[key]
                        # Clean the option content
                        cleaned_content = clean_option_content(option_content, option_letter)
                        if cleaned_content:
                            chinese_entry['options'][option_letter] = cleaned_content
                
                # Extract English fields
                english_entry = {
                    # 'id': len(en_entries) + 1,
                    'abilities': abilities_list,  # Same structured ability for both
                    'story': entry.get('STORY', ''),
                    'index': entry.get('序号\nINDEX', ''),
                    'question': entry.get('QUESTION', ''),
                    'options': {},
                    'answer': entry.get('CORRECT_ANSWER', ''),
                    # 'sheet_name': entry.get('sheet_name', ''),
                    'language': 'English'
                }
                
                # Extract English options
                for key in ['OPTION-A', 'OPTION-B', 'OPTION-C', 'OPTION-D']:
                    if key in entry and entry[key] is not None:
                        # Check if the value is NaN
                        if isinstance(entry[key], float) and math.isnan(entry[key]):
                            continue  # Skip NaN values
                        
                        option_letter = key[-1]  # Extract A, B, C, or D
                        option_content = str(entry[key])  # Convert to string
                        
                         # Clean the option content
                        cleaned_content = clean_option_content(option_content, option_letter)
                        if cleaned_content:
                            english_entry['options'][option_letter] = cleaned_content
                
                # Add both entries
                cn_entries.append(chinese_entry)
                en_entries.append(english_entry)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                continue
    
    return cn_entries, en_entries

def read_all_tombench_files(base_path=os.path.join(BASE_DIR, "raw_data/tombench/data")):
    """
    Read all ToMBench JSONL files and return a combined list of parsed entries
    """
    en_all_entries = []
    cn_all_entries = []
    
    # List of all ToMBench files
    tomb_files = [
        "Ambiguous Story Task.jsonl",
        "Completion of Failed Actions.jsonl",
        "Discrepant Desires.jsonl",
        "Discrepant Emotions.jsonl",
        "Discrepant Intentions.jsonl",
        "Emotion Regulation.jsonl",
        "False Belief Task.jsonl",
        "Faux-pas Recognition Test.jsonl",
        "Hidden Emotions.jsonl",
        "Hinting Task Test.jsonl",
        "Knowledge-Attention Links.jsonl",
        "Knowledge-Pretend Play Links.jsonl",
        "Moral Emotions.jsonl",
        "Multiple Desires.jsonl",
        "Percepts-Knowledge Links.jsonl",
        "Persuasion Story Task.jsonl",
        "Prediction of Actions.jsonl",
        "Scalar Implicature Test.jsonl",
        "Strange Story Task.jsonl",
        "Unexpected Outcome Test.jsonl"
    ]

    # tomb_files = ["False Belief Task.jsonl"]
    
    for filename in tomb_files:
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            print(f"Processing: {filename}")
            try:
                cn_entries, en_entries = parse_tombench_jsonl(file_path)
                cn_all_entries.extend(cn_entries)
                en_all_entries.extend(en_entries)
                print(f"  - Extracted {len(cn_entries)} Chinese entries and {len(en_entries)} English entries")
            except Exception as e:
                print(f"  - Error processing {filename}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nTotal cn entries extracted: {len(cn_all_entries)}")
    print(f"\nTotal en entries extracted: {len(en_all_entries)}")
    return cn_all_entries, en_all_entries

# Example usage
if __name__ == "__main__":
    # Read all ToMBench files
    print("Reading all ToMBench files...")
    cn_all_entries, en_all_entries = read_all_tombench_files()
    
    # Calculate total entries
    total_cn = len(cn_all_entries)
    total_en = len(en_all_entries)
    total_all = total_cn + total_en
    
    # Add ID and total to each entry with global counter
    global_id = 1 
    for entry in cn_all_entries:
        entry['id'] = global_id
        global_id += 1
    
    global_id = 1
    for entry in en_all_entries:
        entry['id'] = global_id
        global_id += 1

    # Print first few entries to verify structure
    for entries in [cn_all_entries, en_all_entries]:
        print("\nFirst 3 entries:")
        for i, entry in enumerate(entries[:3]):
            print(f"\nEntry {i+1}:")
            print(f"Language: {entry['language']}")
            print(f"Ability: {entry['abilities']}")
            print(f"Index: {entry['index']}")
            print(f"Story: {entry['story'][:100]}...")
            print(f"Question: {entry['question']}")
            print(f"Options: {entry['options']}")
            print(f"Answer: {entry['answer']}")
    
    # Save to file
    output_cn_path = os.path.join(BASE_DIR, "raw_data/tombench/parsed/tombench_CN_parsed.json")
    output_en_path = os.path.join(BASE_DIR, "raw_data/tombench/parsed/tombench_EN_parsed.json")
    os.makedirs(os.path.dirname(output_cn_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_en_path), exist_ok=True)
    
    with open(output_cn_path, 'w', encoding='utf-8') as f:
        json.dump(cn_all_entries, f, ensure_ascii=False, indent=2)
    with open(output_en_path, 'w', encoding='utf-8') as f:
        json.dump(en_all_entries, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(cn_all_entries)} Chinese entries and {len(en_all_entries)} English entries to {output_cn_path} and {output_en_path}")
