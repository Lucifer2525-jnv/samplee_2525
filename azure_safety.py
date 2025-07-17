"""
Azure Content Safety and Language Service integration
"""
import os
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

class AzureContentSafety:
    """Azure Content Safety service integration"""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
        self.key = os.getenv("AZURE_CONTENT_SAFETY_KEY")
        self.threshold = os.getenv("CONTENT_SAFETY_THRESHOLD", "Medium")
        self.enabled = bool(self.endpoint and self.key)
        
        # Severity mapping
        self.severity_levels = {
            "Low": 2,
            "Medium": 4,
            "High": 6
        }
        
    def is_enabled(self) -> bool:
        """Check if Content Safety is enabled"""
        return self.enabled
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for harmful content"""
        if not self.enabled:
            return {"safe": True, "categories": {}, "severity": 0}
        
        try:
            url = f"{self.endpoint}/contentsafety/text:analyze?api-version=2023-10-01"
            headers = {
                "Ocp-Apim-Subscription-Key": self.key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "text": text,
                "categories": ["Hate", "SelfHarm", "Sexual", "Violence"],
                "blocklistNames": [],
                "haltOnBlocklistHit": False,
                "outputType": "FourSeverityLevels"
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return self._process_safety_result(result)
            else:
                print(f"Content Safety API error: {response.status_code} - {response.text}")
                return {"safe": True, "categories": {}, "severity": 0, "error": "API_ERROR"}
                
        except Exception as e:
            print(f"Content Safety error: {e}")
            return {"safe": True, "categories": {}, "severity": 0, "error": str(e)}
    
    def _process_safety_result(self, result: Dict) -> Dict[str, Any]:
        """Process Content Safety API result"""
        categories = {}
        max_severity = 0
        threshold_level = self.severity_levels.get(self.threshold, 4)
        
        for category_result in result.get("categoriesAnalysis", []):
            category = category_result["category"]
            severity = category_result["severity"]
            categories[category.lower()] = severity
            max_severity = max(max_severity, severity)
        
        is_safe = max_severity < threshold_level
        
        return {
            "safe": is_safe,
            "categories": categories,
            "severity": max_severity,
            "threshold": threshold_level,
            "blocked_categories": [
                cat for cat, sev in categories.items() 
                if sev >= threshold_level
            ]
        }

class AzureLanguageService:
    """Azure Language Service for PII detection and masking"""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        self.key = os.getenv("AZURE_LANGUAGE_KEY")
        self.pii_enabled = os.getenv("PII_MASKING_ENABLED", "true").lower() == "true"
        self.enabled = bool(self.endpoint and self.key and self.pii_enabled)
        
        # PII categories to detect
        self.pii_categories = [
            "Person", "PersonType", "PhoneNumber", "Email", "URL", 
            "IPAddress", "DateTime", "Quantity", "Organization",
            "Address", "CreditCardNumber", "ABARoutingNumber"
        ]
        
    def is_enabled(self) -> bool:
        """Check if Language Service is enabled"""
        return self.enabled
    
    def detect_and_mask_pii(self, text: str) -> Dict[str, Any]:
        """Detect PII and return masked text"""
        if not self.enabled:
            return {
                "masked_text": text,
                "pii_detected": False,
                "entities": [],
                "original_text": text
            }
        
        try:
            # First detect PII
            pii_result = self._detect_pii(text)
            
            if not pii_result.get("entities"):
                return {
                    "masked_text": text,
                    "pii_detected": False,
                    "entities": [],
                    "original_text": text
                }
            
            # Mask the detected PII
            masked_text = self._mask_pii_entities(text, pii_result["entities"])
            
            return {
                "masked_text": masked_text,
                "pii_detected": True,
                "entities": pii_result["entities"],
                "original_text": text
            }
            
        except Exception as e:
            print(f"Language Service error: {e}")
            return {
                "masked_text": text,
                "pii_detected": False,
                "entities": [],
                "original_text": text,
                "error": str(e)
            }
    
    def _detect_pii(self, text: str) -> Dict[str, Any]:
        """Detect PII entities in text"""
        url = f"{self.endpoint}/language/:analyze-text?api-version=2022-05-01"
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "kind": "PiiEntityRecognition",
            "parameters": {
                "modelVersion": "latest",
                "domain": "phi",  # Protected Health Information domain
                "piiCategories": self.pii_categories
            },
            "analysisInput": {
                "documents": [
                    {
                        "id": "1",
                        "language": "en",
                        "text": text
                    }
                ]
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            documents = result.get("results", {}).get("documents", [])
            if documents:
                return {
                    "entities": documents[0].get("entities", []),
                    "redacted_text": documents[0].get("redactedText", text)
                }
        else:
            print(f"PII Detection API error: {response.status_code} - {response.text}")
        
        return {"entities": [], "redacted_text": text}
    
    def _mask_pii_entities(self, text: str, entities: List[Dict]) -> str:
        """Mask PII entities in text"""
        if not entities:
            return text
        
        # Sort entities by offset in reverse order to avoid index shifting
        sorted_entities = sorted(entities, key=lambda x: x["offset"], reverse=True)
        
        masked_text = text
        for entity in sorted_entities:
            start = entity["offset"]
            end = start + entity["length"]
            category = entity["category"]
            confidence = entity.get("confidenceScore", 0)
            
            # Only mask high-confidence detections
            if confidence >= 0.7:
                mask = self._get_mask_for_category(category, entity.get("text", ""))
                masked_text = masked_text[:start] + mask + masked_text[end:]
        
        return masked_text
    
    def _get_mask_for_category(self, category: str, original_text: str) -> str:
        """Get appropriate mask for PII category"""
        masks = {
            "Person": "[PERSON]",
            "PhoneNumber": "[PHONE]",
            "Email": "[EMAIL]",
            "CreditCardNumber": "[CREDIT_CARD]",
            "Address": "[ADDRESS]",
            "Organization": "[ORGANIZATION]",
            "DateTime": "[DATE]",
            "IPAddress": "[IP_ADDRESS]",
            "URL": "[URL]",
            "ABARoutingNumber": "[ROUTING_NUMBER]"
        }
        
        return masks.get(category, "[REDACTED]")

class SafetyManager:
    """Main safety manager combining both services"""
    
    def __init__(self):
        self.content_safety = AzureContentSafety()
        self.language_service = AzureLanguageService()
        
    def process_user_query(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """Process user query through safety pipeline"""
        result = {
            "original_query": query,
            "processed_query": query,
            "is_safe": True,
            "pii_detected": False,
            "content_blocked": False,
            "safety_analysis": {},
            "pii_analysis": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Step 1: Content Safety Check
            if self.content_safety.is_enabled():
                safety_result = self.content_safety.analyze_text(query)
                result["safety_analysis"] = safety_result
                
                if not safety_result["safe"]:
                    result["is_safe"] = False
                    result["content_blocked"] = True
                    result["blocked_reason"] = f"Content blocked due to: {', '.join(safety_result.get('blocked_categories', []))}"
                    return result
            
            # Step 2: PII Detection and Masking
            if self.language_service.is_enabled():
                pii_result = self.language_service.detect_and_mask_pii(query)
                result["pii_analysis"] = pii_result
                result["pii_detected"] = pii_result["pii_detected"]
                result["processed_query"] = pii_result["masked_text"]
            
            return result
            
        except Exception as e:
            print(f"Safety processing error: {e}")
            result["error"] = str(e)
            return result
    
    def process_llm_response(self, response: str) -> Dict[str, Any]:
        """Process LLM response for safety"""
        result = {
            "original_response": response,
            "processed_response": response,
            "is_safe": True,
            "content_blocked": False,
            "pii_detected": False,
            "safety_analysis": {},
            "pii_analysis": {}
        }
        
        try:
            # Content Safety Check on response
            if self.content_safety.is_enabled():
                safety_result = self.content_safety.analyze_text(response)
                result["safety_analysis"] = safety_result
                
                if not safety_result["safe"]:
                    result["is_safe"] = False
                    result["content_blocked"] = True
                    result["processed_response"] = "I apologize, but I cannot provide that response due to content policy restrictions."
                    return result
            
            # PII Check on response (optional - usually responses shouldn't contain PII)
            if self.language_service.is_enabled():
                pii_result = self.language_service.detect_and_mask_pii(response)
                result["pii_analysis"] = pii_result
                result["pii_detected"] = pii_result["pii_detected"]
                if pii_result["pii_detected"]:
                    result["processed_response"] = pii_result["masked_text"]
            
            return result
            
        except Exception as e:
            print(f"Response safety processing error: {e}")
            result["error"] = str(e)
            return result
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety service status"""
        return {
            "content_safety_enabled": self.content_safety.is_enabled(),
            "pii_detection_enabled": self.language_service.is_enabled(),
            "content_safety_threshold": self.content_safety.threshold,
            "services_status": {
                "content_safety": "enabled" if self.content_safety.is_enabled() else "disabled",
                "language_service": "enabled" if self.language_service.is_enabled() else "disabled"
            }
        }