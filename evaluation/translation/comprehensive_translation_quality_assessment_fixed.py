#!/usr/bin/env python3
"""
Comprehensive Translation Quality Assessment for CoTR - FIXED VERSION

This script implements a reference-based methodology to evaluate models' translations 
against translations created using the NLLB-200 model, which serves as ground truth.

Fixes:
- NLLB tokenizer lang_code_to_id issue
- Language extraction from file paths
- Boolean evaluation errors
- File finding and column matching
"""

import os
import sys
import json
import pandas as pd
import logging
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    from evaluate import load
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.error("Required packages not available. Please install: transformers, torch, evaluate")
    sys.exit(1)

class NLLBTranslator:
    """NLLB-200 translator for ground truth translations - FIXED"""
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        logger.info(f"Loading NLLB model: {model_name}")
        
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
            logger.info("Model moved to GPU")
        else:
            logger.info("GPU not available, using CPU")
        
        # Fixed language code mapping - using flores codes
        self.lang_codes = {
            'sw': 'swh_Latn',  # Swahili
            'ha': 'hau_Latn',  # Hausa
            'pt': 'por_Latn',  # Portuguese
            'fi': 'fin_Latn',  # Finnish
            'en': 'eng_Latn',  # English
            'fr': 'fra_Latn',  # French
            'ur': 'urd_Arab'   # Urdu
        }
        
        # Create reverse mapping for tokenizer
        self.code_to_id = {}
        for lang, flores_code in self.lang_codes.items():
            if hasattr(self.tokenizer, 'lang_code_to_id'):
                if flores_code in self.tokenizer.lang_code_to_id:
                    self.code_to_id[lang] = self.tokenizer.lang_code_to_id[flores_code]
            else:
                # Fallback: find token id manually
                try:
                    token_id = self.tokenizer.convert_tokens_to_ids(flores_code)
                    if token_id != self.tokenizer.unk_token_id:
                        self.code_to_id[lang] = token_id
                except:
                    logger.warning(f"Could not find token ID for {flores_code}")
        
        logger.info(f"NLLB model loaded. Language codes available: {list(self.code_to_id.keys())}")
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using NLLB-200 - FIXED with GPU optimization"""
        if not text or not text.strip():
            return ""
            
        try:
            # Use available language codes or fallback
            src_code = self.lang_codes.get(source_lang, source_lang)
            tgt_code = self.lang_codes.get(target_lang, target_lang)
            
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get target language token ID
            forced_bos_token_id = self.code_to_id.get(target_lang)
            
            # Generate translation with GPU optimization
            with torch.no_grad():
                if forced_bos_token_id:
                    translated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                else:
                    # Fallback without forced BOS token
                    translated_tokens = self.model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
            
            # Decode translation
            translation = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            return translation.strip()
            
        except Exception as e:
            logger.warning(f"Translation failed for '{text[:50]}...': {str(e)}")
            return ""

class BLEUEvaluator:
    """BLEU score calculator using SacreBLEU - FIXED"""
    
    def __init__(self):
        try:
            self.bleu_metric = load("sacrebleu")
            logger.info("SacreBLEU loaded successfully")
        except:
            logger.warning("SacreBLEU not available, using simple fallback")
            self.bleu_metric = None
    
    def calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score between predictions and references - FIXED"""
        if not predictions or not references:
            return 0.0
        
        # Ensure same length
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
        
        # Filter out empty strings
        valid_pairs = [(pred.strip(), ref.strip()) for pred, ref in zip(predictions, references) 
                      if pred.strip() and ref.strip()]
        
        if not valid_pairs:
            return 0.0
        
        valid_predictions, valid_references = zip(*valid_pairs)
        
        try:
            if self.bleu_metric:
                # SacreBLEU expects references as list of lists
                formatted_refs = [[ref] for ref in valid_references]
                result = self.bleu_metric.compute(predictions=list(valid_predictions), references=formatted_refs)
                return result['score'] / 100.0  # Normalize to 0-1
            else:
                return self._simple_bleu_fallback(list(valid_predictions), list(valid_references))
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}, using fallback")
            return self._simple_bleu_fallback(list(valid_predictions), list(valid_references))
    
    def _simple_bleu_fallback(self, predictions: List[str], references: List[str]) -> float:
        """Simple BLEU fallback implementation"""
        if not predictions or not references:
            return 0.0
        
        total_score = 0.0
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            if not pred_words or not ref_words:
                continue
                
            # Simple 1-gram precision
            matches = sum(1 for word in pred_words if word in ref_words)
            precision = matches / len(pred_words) if pred_words else 0.0
            total_score += precision
            
        return total_score / len(predictions) if predictions else 0.0

class ComprehensiveTranslationAssessor:
    """Comprehensive translation quality assessor for all tasks - FIXED"""
    
    def __init__(self):
        self.nllb = NLLBTranslator()
        self.bleu = BLEUEvaluator()
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text - ENHANCED"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove step patterns and artifacts
        text = re.sub(r'Step\s*\d+[:\-].*?(?=Step\s*\d+|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'English\s*Translation[:\-]?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(Swahili|Portuguese|Finnish|Hausa|French|Urdu)\s*Translation[:\-]?', '', text, flags=re.IGNORECASE)
        
        # Clean up artifacts
        text = re.sub(r'```[^`]*```', '', text)
        text = re.sub(r'---+', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'^\s*[\"\']|[\"\']s*$', '', text)  # Remove quotes
        
        return text.strip()
    
    def assess_sentiment_file(self, filepath: Path, lang_code: str) -> Dict[str, Any]:
        """Assess sentiment file with forward translation only - FIXED"""
        logger.info(f"Analyzing Sentiment file: {filepath.name}")
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return self.create_zero_result(filepath, lang_code, 'sentiment')
        
        results = {
            'file': filepath.name,
            'language': lang_code,
            'task': 'sentiment',
            'sample_count': 0,
            'forward_bleu': 0.0,
            'backward_bleu': 0.0,
            'overall_bleu': 0.0,
            'details': {'methodology': 'NLLB_ground_truth_forward_only'}
        }
        
        # Find text columns
        text_col = None
        intermediate_col = None
        
        for col in ['original_text_lrl', 'text', 'input_text']:
            if col in df.columns:
                text_col = col
                break
        
        for col in ['intermediate_en_text', 'intermediate_translation', 'en_translation']:
            if col in df.columns:
                intermediate_col = col
                break
        
        if not text_col or not intermediate_col:
            logger.warning(f"Required columns not found in {filepath.name}. Available: {list(df.columns)}")
            return results
        
        # Filter and process data
        valid_rows = df.dropna(subset=[text_col, intermediate_col])
        
        if len(valid_rows) == 0:
            logger.warning(f"No valid rows found in {filepath.name}")
            return results
        
        lrl_texts = valid_rows[text_col].astype(str).tolist()
        model_en_texts = valid_rows[intermediate_col].astype(str).tolist()
        
        # Clean model outputs
        cleaned_en_texts = [self.clean_text(text) for text in model_en_texts]
        
        # Filter valid pairs
        valid_pairs = [(lrl, clean_en) for lrl, clean_en in zip(lrl_texts, cleaned_en_texts) 
                      if clean_en and len(clean_en.split()) > 1]
        
        if not valid_pairs:
            logger.warning(f"No valid translation pairs in {filepath.name}")
            return results
        
        valid_lrl, valid_clean_en = zip(*valid_pairs)
        
        # Generate NLLB reference translations (LRL → EN)
        logger.info(f"Generating NLLB translations for {len(valid_lrl)} samples")
        nllb_en_texts = []
        
        for text in tqdm(valid_lrl, desc=f"NLLB {lang_code}→EN"):
            translation = self.nllb.translate(text, lang_code, 'en')
            nllb_en_texts.append(translation)
        
        # Calculate forward BLEU
        if nllb_en_texts:
            forward_bleu = self.bleu.calculate_bleu(list(valid_clean_en), nllb_en_texts)
            results['forward_bleu'] = forward_bleu
            results['sample_count'] = len(valid_pairs)
            
            # Backward translation: Model EN -> NLLB LRL -> Compare with original LRL
            backward_bleu = 0.0
            if valid_clean_en:
                logger.info(f"Generating backward translations for {len(valid_clean_en)} samples")
                nllb_back_translations = []
                for en_text in tqdm(valid_clean_en, desc=f"NLLB EN→{lang_code}"):
                    back_trans = self.nllb.translate(en_text, 'en', lang_code)
                    nllb_back_translations.append(self.clean_text(back_trans))
                
                if nllb_back_translations:
                    backward_bleu = self.bleu.calculate_bleu(
                        list(valid_lrl),
                        nllb_back_translations
                    )
                    logger.info(f"Sentiment Backward BLEU: {backward_bleu:.4f}")
            
            results['backward_bleu'] = backward_bleu
            results['overall_bleu'] = (forward_bleu + backward_bleu) / 2
            
            logger.info(f"Sentiment Forward BLEU: {forward_bleu:.4f} ({len(valid_pairs)} samples)")
        
        return results
    
    def assess_classification_file(self, filepath: Path, lang_code: str) -> Dict[str, Any]:
        """Assess translation quality for classification files - FIXED"""
        logger.info(f"Analyzing Classification file: {filepath.name}")
        try:
            df = pd.read_csv(filepath)

            text_col, intermediate_col = None, None
            possible_text_cols = ['text_lrl', 'original_text', 'original_text_lrl']
            possible_intermediate_cols = ['intermediate_en_text', 'text_en_model']

            for col in possible_text_cols:
                if col in df.columns:
                    text_col = col
                    break
            
            for col in possible_intermediate_cols:
                if col in df.columns:
                    intermediate_col = col
                    break

            if not text_col or not intermediate_col:
                logger.warning(f"Required columns not found in {filepath.name}. Available: {list(df.columns)}")
                return self.create_zero_result(filepath, lang_code, "classification")

            # Filter out empty translations
            valid_rows = df.dropna(subset=[text_col, intermediate_col])
            valid_pairs = [(lrl, en) for lrl, en in zip(valid_rows[text_col], valid_rows[intermediate_col])
                          if pd.notna(lrl) and pd.notna(en) and str(lrl).strip() and str(en).strip()]
            
            if not valid_pairs:
                logger.warning(f"No valid text pairs found in {filepath.name}")
                return self.create_zero_result(filepath, lang_code, "classification")
            
            lrl_texts, en_texts = zip(*valid_pairs)
            
            # Generate NLLB ground truth translations
            logger.info(f"Generating NLLB translations for {len(lrl_texts)} samples")
            nllb_translations = []
            cleaned_en_texts = [self.clean_text(text) for text in en_texts]

            for lrl_text in tqdm(lrl_texts, desc=f"NLLB {lang_code}→EN"):
                nllb_trans = self.nllb.translate(lrl_text, lang_code, "en")
                nllb_translations.append(self.clean_text(nllb_trans))
            
            # Calculate forward BLEU (model EN vs NLLB EN)
            forward_bleu = self.bleu.calculate_bleu(cleaned_en_texts, nllb_translations) if nllb_translations else 0.0
            logger.info(f"Classification Forward BLEU: {forward_bleu:.4f}")

            # Calculate backward BLEU
            backward_bleu = 0.0
            if cleaned_en_texts:
                logger.info(f"Generating backward translations for {len(cleaned_en_texts)} samples")
                nllb_back_translations = []
                for en_text in tqdm(cleaned_en_texts, desc=f"NLLB EN→{lang_code}"):
                    back_trans = self.nllb.translate(en_text, 'en', lang_code)
                    nllb_back_translations.append(self.clean_text(back_trans))
                
                if nllb_back_translations:
                    backward_bleu = self.bleu.calculate_bleu(list(lrl_texts), nllb_back_translations)
                    logger.info(f"Classification Backward BLEU: {backward_bleu:.4f}")

            overall_bleu = (forward_bleu + backward_bleu) / 2
            
            return {
                'file': filepath.name,
                'language': lang_code,
                'task': 'classification',
                'sample_count': len(valid_pairs),
                'forward_bleu': forward_bleu,
                'backward_bleu': backward_bleu,
                'overall_bleu': overall_bleu,
                'details': {'methodology': 'NLLB_ground_truth_forward_backward'}
            }
                
        except Exception as e:
            logger.error(f"Error processing classification file {filepath}: {str(e)}")
            return self.create_zero_result(filepath, lang_code, "classification")
    
    def assess_qa_file(self, filepath: Path, lang_code: str) -> Dict[str, Any]:
        """Assess translation quality for QA files - FIXED"""
        try:
            df = pd.read_csv(filepath)
            
            # Check for QA-specific columns
            if 'lrl_question' in df.columns and 'lrl_context' in df.columns and 'en_question_model' in df.columns:
                # QA format - assess question and context translations
                lrl_questions = df['lrl_question'].dropna().tolist()
                lrl_contexts = df['lrl_context'].dropna().tolist()
                en_questions = df['en_question_model'].dropna().tolist()
                en_contexts = df['en_context_model'].dropna().tolist()
                
                # Combine question and context (truncated to 200 chars as per methodology)
                lrl_combined = []
                en_combined = []
                
                for lrl_q, lrl_c, en_q, en_c in zip(lrl_questions, lrl_contexts, en_questions, en_contexts):
                    if pd.notna(lrl_q) and pd.notna(lrl_c) and pd.notna(en_q) and pd.notna(en_c):
                        lrl_text = f"{str(lrl_c)[:200]} {str(lrl_q)}"
                        en_text = f"{str(en_c)[:200]} {str(en_q)}"
                        lrl_combined.append(lrl_text)
                        en_combined.append(en_text)
                
                if not lrl_combined:
                    logger.warning(f"No valid QA pairs found in {filepath.name}")
                    return self.create_zero_result(filepath, lang_code, "qa")
                
                # Generate NLLB ground truth translations
                logger.info(f"Generating NLLB translations for {len(lrl_combined)} QA samples")
                nllb_translations = []
                cleaned_en_combined = [self.clean_text(text) for text in en_combined]
                for lrl_text in tqdm(lrl_combined, desc=f"NLLB QA {lang_code}→EN"):
                    nllb_trans = self.nllb.translate(lrl_text, lang_code, "en")
                    nllb_translations.append(self.clean_text(nllb_trans))
                
                # Calculate forward BLEU (model EN vs NLLB EN)
                forward_bleu = self.bleu.calculate_bleu(cleaned_en_combined, nllb_translations) if nllb_translations else 0.0
                logger.info(f"QA Forward BLEU: {forward_bleu:.4f}")

                # Calculate backward BLEU
                backward_bleu = 0.0
                if cleaned_en_combined:
                    logger.info(f"Generating backward translations for {len(cleaned_en_combined)} samples")
                    nllb_back_translations = []
                    for en_text in tqdm(cleaned_en_combined, desc=f"NLLB EN→{lang_code}"):
                        back_trans = self.nllb.translate(en_text, 'en', lang_code)
                        nllb_back_translations.append(self.clean_text(back_trans))
                    
                    if nllb_back_translations:
                        backward_bleu = self.bleu.calculate_bleu(lrl_combined, nllb_back_translations)
                        logger.info(f"QA Backward BLEU: {backward_bleu:.4f}")

                overall_bleu = (forward_bleu + backward_bleu) / 2
                
                return {
                    'file': filepath.name,
                    'language': lang_code,
                    'task': 'qa',
                    'sample_count': len(lrl_combined),
                    'forward_bleu': forward_bleu,
                    'backward_bleu': backward_bleu,
                    'overall_bleu': overall_bleu,
                    'details': {'methodology': 'NLLB_ground_truth_forward_backward'}
                }
                
            else:
                logger.warning(f"Required QA columns not found in {filepath.name}")
                return self.create_zero_result(filepath, lang_code, "qa")
                
        except Exception as e:
            logger.error(f"Error processing QA file {filepath}: {str(e)}")
            return self.create_zero_result(filepath, lang_code, "qa")
    
    def assess_nli_file(self, filepath: Path, lang_code: str) -> Dict[str, Any]:
        """Assess translation quality for NLI files - FIXED"""
        try:
            df = pd.read_csv(filepath)
            
            # Check for NLI-specific columns
            if 'premise_lrl' in df.columns and 'hypothesis_lrl' in df.columns and 'premise_en' in df.columns and 'hypothesis_en' in df.columns:
                # NLI format - assess premise and hypothesis translations
                lrl_premises = df['premise_lrl'].dropna().tolist()
                lrl_hypotheses = df['hypothesis_lrl'].dropna().tolist()
                en_premises = df['premise_en'].dropna().tolist()
                en_hypotheses = df['hypothesis_en'].dropna().tolist()
                
                # Combine premise and hypothesis
                lrl_combined = []
                en_combined = []
                
                for lrl_p, lrl_h, en_p, en_h in zip(lrl_premises, lrl_hypotheses, en_premises, en_hypotheses):
                    if pd.notna(lrl_p) and pd.notna(lrl_h) and pd.notna(en_p) and pd.notna(en_h):
                        lrl_text = f"{str(lrl_p)} {str(lrl_h)}"
                        en_text = f"{str(en_p)} {str(en_h)}"
                        lrl_combined.append(lrl_text)
                        en_combined.append(en_text)
                
                if not lrl_combined:
                    logger.warning(f"No valid NLI pairs found in {filepath.name}")
                    return self.create_zero_result(filepath, lang_code, "nli")
                
                # Generate NLLB ground truth translations
                logger.info(f"Generating NLLB translations for {len(lrl_combined)} NLI samples")
                nllb_translations = []
                cleaned_en_combined = [self.clean_text(text) for text in en_combined]
                for lrl_text in tqdm(lrl_combined, desc=f"NLLB NLI {lang_code}→EN"):
                    nllb_trans = self.nllb.translate(lrl_text, lang_code, "en")
                    nllb_translations.append(self.clean_text(nllb_trans))
                
                # Calculate forward BLEU (model EN vs NLLB EN)
                forward_bleu = self.bleu.calculate_bleu(cleaned_en_combined, nllb_translations) if nllb_translations else 0.0
                logger.info(f"NLI Forward BLEU: {forward_bleu:.4f}")

                # Calculate backward BLEU
                backward_bleu = 0.0
                if cleaned_en_combined:
                    logger.info(f"Generating backward translations for {len(cleaned_en_combined)} samples")
                    nllb_back_translations = []
                    for en_text in tqdm(cleaned_en_combined, desc=f"NLLB EN→{lang_code}"):
                        back_trans = self.nllb.translate(en_text, 'en', lang_code)
                        nllb_back_translations.append(self.clean_text(back_trans))
                    
                    if nllb_back_translations:
                        backward_bleu = self.bleu.calculate_bleu(lrl_combined, nllb_back_translations)
                        logger.info(f"NLI Backward BLEU: {backward_bleu:.4f}")

                overall_bleu = (forward_bleu + backward_bleu) / 2
                
                return {
                    'file': filepath.name,
                    'language': lang_code,
                    'task': 'nli',
                    'sample_count': len(lrl_combined),
                    'forward_bleu': forward_bleu,
                    'backward_bleu': backward_bleu,
                    'overall_bleu': overall_bleu,
                    'details': {'methodology': 'NLLB_ground_truth_forward_backward'}
                }
                
            else:
                logger.warning(f"Required NLI columns not found in {filepath.name}")
                return self.create_zero_result(filepath, lang_code, "nli")
                
        except Exception as e:
            logger.error(f"Error processing NLI file {filepath}: {str(e)}")
            return self.create_zero_result(filepath, lang_code, "nli")
    
    def assess_ner_file(self, filepath: Path, lang_code: str) -> Dict[str, Any]:
        """Assess NER file - FIXED"""
        logger.info(f"Analyzing NER file: {filepath.name}")
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return self.create_zero_result(filepath, lang_code, 'ner')
        
        results = {
            'file': filepath.name,
            'language': lang_code,
            'task': 'ner',
            'sample_count': len(df),
            'forward_bleu': 0.0,
            'backward_bleu': 0.0,
            'overall_bleu': 0.0,
            'details': {'methodology': 'NLLB_ground_truth_forward_backward'}
        }
        
        # Find text columns
        text_col = None
        intermediate_col = None
        
        for col in ['original_text_lrl', 'text', 'input_text']:
            if col in df.columns:
                text_col = col
                break
        
        for col in ['intermediate_en_text', 'intermediate_translation', 'en_translation']:
            if col in df.columns:
                intermediate_col = col
                break
        
        if not text_col or not intermediate_col:
            logger.warning(f"Required columns not found in {filepath.name}. Available: {list(df.columns)}")
            return results
        
        # Process text translation
        valid_rows = df.dropna(subset=[text_col, intermediate_col])
        
        if len(valid_rows) > 0:
            lrl_texts = valid_rows[text_col].astype(str).tolist()
            model_en_texts = valid_rows[intermediate_col].astype(str).tolist()
            
            # Clean model outputs
            cleaned_en_texts = [self.clean_text(text) for text in model_en_texts]
            
            # Filter valid pairs
            valid_pairs = [(lrl, clean_en) for lrl, clean_en in zip(lrl_texts, cleaned_en_texts) 
                          if clean_en and len(clean_en.split()) > 1]
            
            if valid_pairs:
                valid_lrl_texts, valid_clean_en_texts = zip(*valid_pairs)
                
                # Generate NLLB reference translations
                nllb_en_texts = []
                for text in tqdm(valid_lrl_texts, desc=f"NLLB Text {lang_code}→EN"):
                    translation = self.nllb.translate(text, lang_code, 'en')
                    nllb_en_texts.append(translation)
                
                if nllb_en_texts:
                    text_bleu = self.bleu.calculate_bleu(list(valid_clean_en_texts), nllb_en_texts)
                    results['forward_bleu'] = text_bleu
                    
                    # Add backward translation for NER
                    backward_bleu = 0.0
                    if valid_clean_en_texts:
                        logger.info(f"Generating backward translations for {len(valid_clean_en_texts)} NER samples")
                        nllb_back_translations = []
                        for en_text in tqdm(valid_clean_en_texts, desc=f"NLLB EN→{lang_code}"):
                            back_trans = self.nllb.translate(en_text, 'en', lang_code)
                            nllb_back_translations.append(self.clean_text(back_trans))
                        
                        if nllb_back_translations:
                            backward_bleu = self.bleu.calculate_bleu(list(valid_lrl_texts), nllb_back_translations)
                            logger.info(f"NER Backward BLEU: {backward_bleu:.4f}")
                    
                    results['backward_bleu'] = backward_bleu
                    results['overall_bleu'] = (text_bleu + backward_bleu) / 2
                    
                    logger.info(f"NER Forward BLEU: {text_bleu:.4f}")
        
        return results
    
    def create_zero_result(self, filepath: Path, lang_code: str, task: str) -> Dict[str, Any]:
        """Create zero result for failed files"""
        return {
            'file': filepath.name,
            'language': lang_code,
            'task': task,
            'sample_count': 0,
            'forward_bleu': 0.0,
            'backward_bleu': 0.0,
            'overall_bleu': 0.0,
            'details': {'error': 'Failed to process'}
        }

def extract_language_from_path(filepath: Path) -> Optional[str]:
    """Extract language code from file path - ENHANCED"""
    path_str = str(filepath).lower()
    
    # Enhanced language patterns
    lang_patterns = {
        'sw': ['_sw_', '_sw.', '/sw/', '_swahili_'],
        'ha': ['_ha_', '_ha.', '/ha/', '_hausa_', '_hau_'],
        'pt': ['_pt_', '_pt.', '/pt/', '_portuguese_', '_por_'],
        'fi': ['_fi_', '_fi.', '/fi/', '_finnish_', '_fin_'],
        'fr': ['_fr_', '_fr.', '/fr/', '_french_', '_fra_'],
        'ur': ['_ur_', '_ur.', '/ur/', '_urdu_', '_urd_']
    }
    
    for lang_code, patterns in lang_patterns.items():
        for pattern in patterns:
            if pattern in path_str:
                return lang_code
    
    return None

def find_all_cotr_files() -> List[Path]:
    """Find ALL CoTR files using the provided file paths"""
    
    # All file paths from user
    file_paths = [
        # Sentiment files
        "results/sentiment_new/cotr/results/single_prompt/zs/sw/Qwen2.5-7B-Instruct/results_sentiment_cotr_sp_zs_sw_Qwen2.5-7B-Instruct.csv",
        "results/sentiment_new/cotr/results/single_prompt/zs/sw/aya-23-8B/results_sentiment_cotr_sp_zs_sw_aya-23-8B.csv",
        "results/sentiment_new/cotr/results/single_prompt/zs/pt/Qwen2.5-7B-Instruct/results_sentiment_cotr_sp_zs_pt_Qwen2.5-7B-Instruct.csv",
        "results/sentiment_new/cotr/results/single_prompt/zs/pt/aya-23-8B/results_sentiment_cotr_sp_zs_pt_aya-23-8B.csv",
        "results/sentiment_new/cotr/results/single_prompt/zs/ha/Qwen2.5-7B-Instruct/results_sentiment_cotr_sp_zs_ha_Qwen2.5-7B-Instruct.csv",
        "results/sentiment_new/cotr/results/single_prompt/zs/ha/aya-23-8B/results_sentiment_cotr_sp_zs_ha_aya-23-8B.csv",
        "results/sentiment_new/cotr/results/single_prompt/fs/sw/Qwen2.5-7B-Instruct/results_sentiment_cotr_sp_fs_sw_Qwen2.5-7B-Instruct.csv",
        "results/sentiment_new/cotr/results/single_prompt/fs/sw/aya-23-8B/results_sentiment_cotr_sp_fs_sw_aya-23-8B.csv",
        "results/sentiment_new/cotr/results/single_prompt/fs/pt/Qwen2.5-7B-Instruct/results_sentiment_cotr_sp_fs_pt_Qwen2.5-7B-Instruct.csv",
        "results/sentiment_new/cotr/results/single_prompt/fs/pt/aya-23-8B/results_sentiment_cotr_sp_fs_pt_aya-23-8B.csv",
        "results/sentiment_new/cotr/results/single_prompt/fs/ha/Qwen2.5-7B-Instruct/results_sentiment_cotr_sp_fs_ha_Qwen2.5-7B-Instruct.csv",
        "results/sentiment_new/cotr/results/single_prompt/fs/ha/aya-23-8B/results_sentiment_cotr_sp_fs_ha_aya-23-8B.csv",
        "results/sentiment_new/cotr/results/multi_prompt/zs/sw/Qwen2.5-7B-Instruct/results_sentiment_cotr_mp_zs_sw_Qwen2.5-7B-Instruct.csv",
        "results/sentiment_new/cotr/results/multi_prompt/zs/sw/aya-23-8B/results_sentiment_cotr_mp_zs_sw_aya-23-8B.csv",
        "results/sentiment_new/cotr/results/multi_prompt/zs/pt/Qwen2.5-7B-Instruct/results_sentiment_cotr_mp_zs_pt_Qwen2.5-7B-Instruct.csv",
        "results/sentiment_new/cotr/results/multi_prompt/zs/pt/aya-23-8B/results_sentiment_cotr_mp_zs_pt_aya-23-8B.csv",
        "results/sentiment_new/cotr/results/multi_prompt/zs/ha/Qwen2.5-7B-Instruct/results_sentiment_cotr_mp_zs_ha_Qwen2.5-7B-Instruct.csv",
        "results/sentiment_new/cotr/results/multi_prompt/zs/ha/aya-23-8B/results_sentiment_cotr_mp_zs_ha_aya-23-8B.csv",
        "results/sentiment_new/cotr/results/multi_prompt/fs/sw/Qwen2.5-7B-Instruct/results_sentiment_cotr_mp_fs_sw_Qwen2.5-7B-Instruct.csv",
        "results/sentiment_new/cotr/results/multi_prompt/fs/sw/aya-23-8B/results_sentiment_cotr_mp_fs_sw_aya-23-8B.csv",
        "results/sentiment_new/cotr/results/multi_prompt/fs/pt/Qwen2.5-7B-Instruct/results_sentiment_cotr_mp_fs_pt_Qwen2.5-7B-Instruct.csv",
        "results/sentiment_new/cotr/results/multi_prompt/fs/pt/aya-23-8B/results_sentiment_cotr_mp_fs_pt_aya-23-8B.csv",
        "results/sentiment_new/cotr/results/multi_prompt/fs/ha/Qwen2.5-7B-Instruct/results_sentiment_cotr_mp_fs_ha_Qwen2.5-7B-Instruct.csv",
        "results/sentiment_new/cotr/results/multi_prompt/fs/ha/aya-23-8B/results_sentiment_cotr_mp_fs_ha_aya-23-8B.csv",
        
        # QA files
        "results/qa_new/cotr/results_cotr_mp_fs_qa_tydiqa_fi_aya-23-8B.csv",
        "results/qa_new/cotr/results_cotr_mp_fs_qa_tydiqa_fi_Qwen2.5-7B-Instruct.csv",
        "results/qa_new/cotr/results_cotr_mp_fs_qa_tydiqa_sw_aya-23-8B.csv",
        "results/qa_new/cotr/results_cotr_mp_fs_qa_tydiqa_sw_Qwen2.5-7B-Instruct.csv",
        "results/qa_new/cotr/results_cotr_mp_zs_qa_tydiqa_fi_aya-23-8B.csv",
        "results/qa_new/cotr/results_cotr_mp_zs_qa_tydiqa_fi_Qwen2.5-7B-Instruct.csv",
        "results/qa_new/cotr/results_cotr_mp_zs_qa_tydiqa_sw_aya-23-8B.csv",
        "results/qa_new/cotr/results_cotr_mp_zs_qa_tydiqa_sw_Qwen2.5-7B-Instruct.csv",
        "results/qa_new/cotr/results_cotr_sp_fs_qa_tydiqa_fi_aya-23-8B.csv",
        "results/qa_new/cotr/results_cotr_sp_fs_qa_tydiqa_fi_Qwen2.5-7B-Instruct.csv",
        "results/qa_new/cotr/results_cotr_sp_fs_qa_tydiqa_sw_aya-23-8B.csv",
        "results/qa_new/cotr/results_cotr_sp_fs_qa_tydiqa_sw_Qwen2.5-7B-Instruct.csv",
        "results/qa_new/cotr/results_cotr_sp_zs_qa_tydiqa_fi_aya-23-8B.csv",
        "results/qa_new/cotr/results_cotr_sp_zs_qa_tydiqa_fi_Qwen2.5-7B-Instruct.csv",
        "results/qa_new/cotr/results_cotr_sp_zs_qa_tydiqa_sw_aya-23-8B.csv",
        "results/qa_new/cotr/results_cotr_sp_zs_qa_tydiqa_sw_Qwen2.5-7B-Instruct.csv",
        
        # NLI files
        "results/nli_new/cotr/single_prompt/zs/ur/Qwen2_5_7B_Instruct/results_cotr_single_prompt_zs_nli_ur_Qwen2_5_7B_Instruct.csv",
        "results/nli_new/cotr/single_prompt/zs/ur/aya_23_8B/results_cotr_single_prompt_zs_nli_ur_aya_23_8B.csv",
        "results/nli_new/cotr/single_prompt/zs/sw/Qwen2_5_7B_Instruct/results_cotr_single_prompt_zs_nli_sw_Qwen2_5_7B_Instruct.csv",
        "results/nli_new/cotr/single_prompt/zs/sw/aya_23_8B/results_cotr_single_prompt_zs_nli_sw_aya_23_8B.csv",
        "results/nli_new/cotr/single_prompt/zs/fr/Qwen2_5_7B_Instruct/results_cotr_single_prompt_zs_nli_fr_Qwen2_5_7B_Instruct.csv",
        "results/nli_new/cotr/single_prompt/zs/fr/aya_23_8B/results_cotr_single_prompt_zs_nli_fr_aya_23_8B.csv",
        "results/nli_new/cotr/single_prompt/fs/ur/Qwen2_5_7B_Instruct/results_cotr_single_prompt_fs_nli_ur_Qwen2_5_7B_Instruct.csv",
        "results/nli_new/cotr/single_prompt/fs/ur/aya_23_8B/results_cotr_single_prompt_fs_nli_ur_aya_23_8B.csv",
        "results/nli_new/cotr/single_prompt/fs/sw/Qwen2_5_7B_Instruct/results_cotr_single_prompt_fs_nli_sw_Qwen2_5_7B_Instruct.csv",
        "results/nli_new/cotr/single_prompt/fs/sw/aya_23_8B/results_cotr_single_prompt_fs_nli_sw_aya_23_8B.csv",
        "results/nli_new/cotr/single_prompt/fs/fr/Qwen2_5_7B_Instruct/results_cotr_single_prompt_fs_nli_fr_Qwen2_5_7B_Instruct.csv",
        "results/nli_new/cotr/single_prompt/fs/fr/aya_23_8B/results_cotr_single_prompt_fs_nli_fr_aya_23_8B.csv",
        "results/nli_new/cotr/multi_prompt/zs/ur/Qwen2_5_7B_Instruct/results_cotr_multi_prompt_zs_nli_ur_Qwen2_5_7B_Instruct.csv",
        "results/nli_new/cotr/multi_prompt/zs/ur/aya_23_8B/results_cotr_multi_prompt_zs_nli_ur_aya_23_8B.csv",
        "results/nli_new/cotr/multi_prompt/zs/sw/Qwen2_5_7B_Instruct/results_cotr_multi_prompt_zs_nli_sw_Qwen2_5_7B_Instruct.csv",
        "results/nli_new/cotr/multi_prompt/zs/sw/aya_23_8B/results_cotr_multi_prompt_zs_nli_sw_aya_23_8B.csv",
        "results/nli_new/cotr/multi_prompt/zs/fr/Qwen2_5_7B_Instruct/results_cotr_multi_prompt_zs_nli_fr_Qwen2_5_7B_Instruct.csv",
        "results/nli_new/cotr/multi_prompt/zs/fr/aya_23_8B/results_cotr_multi_prompt_zs_nli_fr_aya_23_8B.csv",
        "results/nli_new/cotr/multi_prompt/fs/ur/Qwen2_5_7B_Instruct/results_cotr_multi_prompt_fs_nli_ur_Qwen2_5_7B_Instruct.csv",
        "results/nli_new/cotr/multi_prompt/fs/ur/aya_23_8B/results_cotr_multi_prompt_fs_nli_ur_aya_23_8B.csv",
        "results/nli_new/cotr/multi_prompt/fs/sw/Qwen2_5_7B_Instruct/results_cotr_multi_prompt_fs_nli_sw_Qwen2_5_7B_Instruct.csv",
        "results/nli_new/cotr/multi_prompt/fs/sw/aya_23_8B/results_cotr_multi_prompt_fs_nli_sw_aya_23_8B.csv",
        "results/nli_new/cotr/multi_prompt/fs/fr/Qwen2_5_7B_Instruct/results_cotr_multi_prompt_fs_nli_fr_Qwen2_5_7B_Instruct.csv",
        "results/nli_new/cotr/multi_prompt/fs/fr/aya_23_8B/results_cotr_multi_prompt_fs_nli_fr_aya_23_8B.csv",
        
        # NER files
        "results/ner_new/cotr/results_cotr_mp_fs_ner_ha_aya-23-8B.csv",
        "results/ner_new/cotr/results_cotr_mp_fs_ner_hau_Qwen2.5-7B-Instruct.csv",
        "results/ner_new/cotr/results_cotr_mp_fs_ner_sw_aya-23-8B.csv",
        "results/ner_new/cotr/results_cotr_mp_fs_ner_sw_Qwen2.5-7B-Instruct.csv",
        "results/ner_new/cotr/results_cotr_mp_zs_ner_ha_aya-23-8B.csv",
        "results/ner_new/cotr/results_cotr_mp_zs_ner_ha_Qwen2.5-7B-Instruct.csv",
        "results/ner_new/cotr/results_cotr_mp_zs_ner_sw_aya-23-8B.csv",
        "results/ner_new/cotr/results_cotr_mp_zs_ner_sw_Qwen2.5-7B-Instruct.csv",
        "results/ner_new/cotr/results_cotr_sp_fs_ner_ha_aya-23-8B.csv",
        "results/ner_new/cotr/results_cotr_sp_fs_ner_hau_Qwen2.5-7B-Instruct.csv",
        "results/ner_new/cotr/results_cotr_sp_fs_ner_sw_aya-23-8B.csv",
        "results/ner_new/cotr/results_cotr_sp_fs_ner_sw_Qwen2.5-7B-Instruct.csv",
        "results/ner_new/cotr/results_cotr_sp_zs_ner_ha_aya-23-8B.csv",
        "results/ner_new/cotr/results_cotr_sp_zs_ner_hau_Qwen2.5-7B-Instruct.csv",
        "results/ner_new/cotr/results_cotr_sp_zs_ner_sw_aya-23-8B.csv",
        "results/ner_new/cotr/results_cotr_sp_zs_ner_sw_Qwen2.5-7B-Instruct.csv",
        
        # Classification files
        "results/classification_new/cotr/multi_prompt/fs/ha/aya-23-8B/results_cotr_classification_ha.csv",
        "results/classification_new/cotr/multi_prompt/fs/ha/Qwen2.5-7B-Instruct/results_cotr_classification_ha.csv",
        "results/classification_new/cotr/multi_prompt/fs/sw/aya-23-8B/results_cotr_classification_sw.csv",
        "results/classification_new/cotr/multi_prompt/fs/sw/Qwen2.5-7B-Instruct/results_cotr_classification_sw.csv",
        "results/classification_new/cotr/multi_prompt/zs/ha/aya-23-8B/results_cotr_classification_ha.csv",
        "results/classification_new/cotr/multi_prompt/zs/ha/Qwen2.5-7B-Instruct/results_cotr_classification_ha.csv",
        "results/classification_new/cotr/multi_prompt/zs/sw/aya-23-8B/results_cotr_classification_sw.csv",
        "results/classification_new/cotr/multi_prompt/zs/sw/Qwen2.5-7B-Instruct/results_cotr_classification_sw.csv",
        "results/classification_new/cotr/single_prompt/fs/ha/aya-23-8B/results_cotr_classification_ha.csv",
        "results/classification_new/cotr/single_prompt/fs/ha/Qwen2.5-7B-Instruct/results_cotr_classification_ha.csv",
        "results/classification_new/cotr/single_prompt/fs/sw/aya-23-8B/results_cotr_classification_sw.csv",
        "results/classification_new/cotr/single_prompt/fs/sw/Qwen2.5-7B-Instruct/results_cotr_classification_sw.csv",
        "results/classification_new/cotr/single_prompt/zs/ha/aya-23-8B/results_cotr_classification_ha.csv",
        "results/classification_new/cotr/single_prompt/zs/ha/Qwen2.5-7B-Instruct/results_cotr_classification_ha.csv",
        "results/classification_new/cotr/single_prompt/zs/sw/aya-23-8B/results_cotr_classification_sw.csv",
        "results/classification_new/cotr/single_prompt/zs/sw/Qwen2.5-7B-Instruct/results_cotr_classification_sw.csv"
    ]
    
    # Convert to Path objects and filter existing files
    existing_files = []
    for path_str in file_paths:
        path = Path(path_str)
        if path.exists():
            existing_files.append(path)
        else:
            logger.warning(f"File not found: {path}")
    
    logger.info(f"Found {len(existing_files)} existing files out of {len(file_paths)} specified")
    return existing_files

def main():
    """Main function to run comprehensive translation quality assessment - FIXED"""
    logger.info("Starting Comprehensive Translation Quality Assessment - FIXED VERSION")
    
    # Initialize assessor
    assessor = ComprehensiveTranslationAssessor()
    
    # Find all files using explicit paths
    all_files = find_all_cotr_files()
    
    if not all_files:
        logger.error("No files found to process!")
        return
    
    # Organize files by task
    task_files = {
        'sentiment': [],
        'classification': [],
        'qa': [],
        'nli': [],
        'ner': []
    }
    
    for filepath in all_files:
        path_str = str(filepath).lower()
        if 'sentiment' in path_str:
            task_files['sentiment'].append(filepath)
        elif 'classification' in path_str:
            task_files['classification'].append(filepath)
        elif 'qa' in path_str:
            task_files['qa'].append(filepath)
        elif 'nli' in path_str:
            task_files['nli'].append(filepath)
        elif 'ner' in path_str:
            task_files['ner'].append(filepath)
    
    # Task-specific analyzers
    task_analyzers = {
        'sentiment': assessor.assess_sentiment_file,
        'classification': assessor.assess_classification_file,
        'qa': assessor.assess_qa_file,
        'nli': assessor.assess_nli_file,
        'ner': assessor.assess_ner_file
    }
    
    all_results = []
    task_summaries = {}
    
    for task, files in task_files.items():
        if not files:
            logger.warning(f"No files found for task: {task}")
            continue
        
        logger.info(f"\n{'='*50}")
        logger.info(f"ANALYZING {task.upper()} TASK ({len(files)} files)")
        logger.info(f"{'='*50}")
        
        task_results = []
        analyzer = task_analyzers[task]
        
        for filepath in files:
            # Extract language
            lang_code = extract_language_from_path(filepath)
            if not lang_code:
                logger.warning(f"Could not extract language from {filepath}")
                continue
            
            try:
                result = analyzer(filepath, lang_code)
                task_results.append(result)
                all_results.append(result)
                logger.info(f"  ✓ {filepath.name} ({lang_code}): "
                           f"Forward={result['forward_bleu']:.4f}, "
                           f"Backward={result['backward_bleu']:.4f}, "
                           f"Overall={result['overall_bleu']:.4f}")
            except Exception as e:
                logger.error(f"  ✗ Error analyzing {filepath.name}: {e}")
        
        # Calculate task summary
        if task_results:
            task_forward_scores = [r['forward_bleu'] for r in task_results if r['forward_bleu'] > 0]
            task_backward_scores = [r['backward_bleu'] for r in task_results if r['backward_bleu'] > 0]
            task_overall_scores = [r['overall_bleu'] for r in task_results if r['overall_bleu'] > 0]
            
            task_summaries[task] = {
                'file_count': len(task_results),
                'avg_forward_bleu': np.mean(task_forward_scores) if task_forward_scores else 0.0,
                'avg_backward_bleu': np.mean(task_backward_scores) if task_backward_scores else 0.0,
                'avg_overall_bleu': np.mean(task_overall_scores) if task_overall_scores else 0.0
            }
            
            logger.info(f"\n{task.upper()} SUMMARY:")
            logger.info(f"  Files processed: {len(task_results)}")
            logger.info(f"  Average Forward BLEU: {task_summaries[task]['avg_forward_bleu']:.4f}")
            logger.info(f"  Average Backward BLEU: {task_summaries[task]['avg_backward_bleu']:.4f}")
            logger.info(f"  Average Overall BLEU: {task_summaries[task]['avg_overall_bleu']:.4f}")
    
    # Calculate global summary
    if all_results:
        global_forward_scores = [r['forward_bleu'] for r in all_results if r['forward_bleu'] > 0]
        global_backward_scores = [r['backward_bleu'] for r in all_results if r['backward_bleu'] > 0]
        global_overall_scores = [r['overall_bleu'] for r in all_results if r['overall_bleu'] > 0]
        
        global_summary = {
            'total_files': len(all_results),
            'avg_forward_bleu': np.mean(global_forward_scores) if global_forward_scores else 0.0,
            'avg_backward_bleu': np.mean(global_backward_scores) if global_backward_scores else 0.0,
            'avg_overall_bleu': np.mean(global_overall_scores) if global_overall_scores else 0.0,
            'methodology': 'NLLB_ground_truth_reference_based_FIXED'
        }
    else:
        global_summary = {'total_files': 0, 'avg_forward_bleu': 0.0, 'avg_backward_bleu': 0.0, 'avg_overall_bleu': 0.0}
    
    # Save results
    output_dir = Path("evaluation/translation/comprehensive_assessment_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    detailed_results = {
        'summary': global_summary,
        'task_summaries': task_summaries,
        'detailed_results': all_results
    }
    
    with open(output_dir / "comprehensive_translation_assessment_fixed.json", 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Save CSV summary
    csv_results = []
    for result in all_results:
        csv_results.append({
            'task': result.get('task'),
            'file': result.get('file'),
            'language': result.get('language'),
            'sample_count': result.get('sample_count', result.get('samples_processed', 0)),
            'forward_bleu': result.get('forward_bleu'),
            'backward_bleu': result.get('backward_bleu'),
            'overall_bleu': result.get('overall_bleu')
        })
    
    df_results = pd.DataFrame(csv_results)
    df_results.to_csv(output_dir / "comprehensive_translation_assessment_fixed.csv", index=False)
    
    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info("COMPREHENSIVE TRANSLATION QUALITY ASSESSMENT COMPLETE - FIXED")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {global_summary['total_files']}")
    logger.info(f"Average Forward BLEU: {global_summary['avg_forward_bleu']:.4f}")
    logger.info(f"Average Backward BLEU: {global_summary['avg_backward_bleu']:.4f}")
    logger.info(f"Average Overall BLEU: {global_summary['avg_overall_bleu']:.4f}")
    logger.info(f"\nResults saved to: {output_dir}")
    
    return detailed_results

if __name__ == "__main__":
    main() 