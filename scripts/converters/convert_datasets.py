#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from convert_csbench import convert as conv_cs
from convert_wemath import convert as conv_wm
from convert_medxpertqa import convert as conv_med
from convert_tombench import convert as conv_tom

from pathlib import Path



def convert_csbench(base):
    for lang in ("CN", "EN", "DE", "FR"):
        for split in ("valid", "test"):
            in_path  = f"{base}/raw_data/csbench/Dataset/CSBench-{lang}/clean_{split}.json"
            out_path = f"{base}/input_data/csbench/csbench_{lang}_{split}_input.jsonl"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if not os.path.exists(in_path):
                print(f"[CSBench] missing {in_path}, skipped.")
                continue
            print(f"[CSBench] {lang}/{split}: {in_path} -> {out_path}")
            conv_cs(in_path, out_path)


def convert_wemath(base):
    in_path  = f"{base}/raw_data/wemath/testmini.json"
    out_path = f"{base}/input_data/wemath/wemath_input.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(in_path):
        print(f"[We-Math] missing {in_path}, skipped.")
        return
    print(f"[We-Math] {in_path} -> {out_path}")
    conv_wm(in_path, out_path)

def convert_medxpertqa(base):
    for modal in ("mm", "text"):
        in_path  = f"{base}/raw_data/medxpertqa/input/medxpertqa_{modal}_input.jsonl"
        out_path = f"{base}/input_data/medxpertqa/medxpertqa_{modal}_input.jsonl"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not os.path.exists(in_path):
            print(f"[MedXpertQA] missing {in_path}, skipped.")
            continue
        print(f"[MedXpertQA] {modal}: {in_path} -> {out_path}")
        conv_med(in_path, out_path)

def convert_tombench(base):
    for lang_code in ("CN", "EN"):
        in_path  = f"{base}/raw_data/tombench/parsed/tombench_{lang_code}_parsed.json"
        out_path = f"{base}/input_data/tombench/tombench_{lang_code}_input.jsonl"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not os.path.exists(in_path):
            print(f"[ToMBench] missing {in_path}, skipped.")
            continue
        print(f"[ToMBench] {lang_code}: {in_path} -> {out_path}")
        conv_tom(in_path, out_path)

def main():

    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parents[1]
    DATA_DIR = f"{BASE_DIR}/data"
    RAW_DATA_DIR = f"{DATA_DIR}/raw_data"
    INPUT_DATA_DIR = f"{DATA_DIR}/input_data"

    # print(BASE_DIR, DATA_DIR, RAW_DATA_DIR, INPUT_DATA_DIR)

    p = argparse.ArgumentParser(description="Convert raw/parsed datasets to unified pipeline inputs.")
    p.add_argument("--base", default=f"{DATA_DIR}",
                   help="Base data directory containing raw_data/ and input_data/ subfolders")
    p.add_argument("--only", nargs="+", choices=["csbench","wemath","med","tombench"],
                   help="Only convert selected dataset(s). Default: all.")
    args = p.parse_args()
    base = args.base

    os.makedirs(os.path.join(base,"input_data"), exist_ok=True)

    # Decide which converters to run
    selected = set(args.only) if args.only else {"csbench","wemath","med","tombench"}
    
    if "csbench" in selected: convert_csbench(base)
    if "wemath" in selected: convert_wemath(base)
    if "med"     in selected: convert_medxpertqa(base)
    if "tombench"    in selected: convert_tombench(base)

if __name__ == "__main__":
    main()
