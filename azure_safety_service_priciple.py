"""
Azure Content Safety and Language Service integration with Service Principal authentication
"""
import os
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import aiohttp
from azure.identity.aio import ClientSecretCredential
from dotenv import load_dotenv

load_dotenv()

class AzureContentSafety:
    """Azure Content Safety service integration with Service Principal auth"""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
        self.tenant_id = os.getenv("TENANT_ID")
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        
        # Content safety thresholds
        self.hate_threshold = int(os.getenv("CONTENT_SAFETY_HATE_THRESHOLD", "4"))
        self.violence_threshold = int(os.getenv("CONTENT_SAFETY_VIOLENCE_THRESHOLD", "4"))
        self.sexual_threshold = int(os.getenv("CONTENT_SAFETY_SEXUAL_THRESHOLD", "4"))
        self.self_harm_threshold = int(os.getenv("CONTENT_SAFETY_SELF_HARM_THRESHOLD", "4"))
        
        self.enabled = bool(self.endpoint and self.tenant_id and self.client_id and self.client_secret)
        self.credential = None
        self.access_token = None
        self.token_expires_at = None
        
        if self.enabled:
            self.credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
        
    def is_enabled(self) -> bool:
        """Check if Content Safety is enabled"""
        return self.enabled
    
    async def _get_access_token(self) -> str:
        """Get access token using Service Principal"""
        if not self.enabled:
            raise Exception("Azure Content Safety not properly configured")
        
        try:
            # Azure Cognitive Services scope
            scope = "https://cognitiveservices.azure.com/.default"
            token = await self.credential.get_token(scope)
            return token.token
        except Exception as e:
            print(f"Error getting access token for Content Safety: {e}")
            raise
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for harmful content"""
        if not self.enabled:
            return {"safe": True, "categories": {}, "severity": 0}
        
        try:
            access_token = await self._get_access_token()
            url = f"{self.endpoint}/contentsafety/text:analyze?api-version=2023-10-01"
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "text": text,
                "categories": ["Hate", "SelfHarm", "Sexual", "Violence"],
                "blocklistNames": [],
                "haltOnBlocklistHit": False,
                "outputType": "FourSeverityLevels"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=10) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._process_safety_result(result)
                    else:
                        error_text = await response.text()
                        print(f"Content Safety API error: {response.status} - {error_text}")
                        return {"safe": True, "categories": {}, "severity": 0, "error": "API_ERROR"}
                        
        except Exception as e:
            print(f"Content Safety error: {e}")
            return {"safe": True, "categories": {}, "severity": 0, "error": str(e)}
    
    def _process_safety_result(self, result: Dict) -> Dict[str, Any]:
        """Process Content Safety API result"""
        categories = {}
        max_severity = 0
        blocked_categories = []
        
        thresholds = {
            "hate": self.hate_threshold,
            "violence": self.violence_threshold,
            "sexual": self.sexual_threshold,
            "selfharm": self.self_harm_threshold
        }
        
        for category_result in result.get("categoriesAnalysis", []):
            category = category_result["category"].lower()
            severity = category_result["severity"]
            categories[category] = severity
            max_severity = max(max_severity, severity)
            
            # Check if this category exceeds threshold
            threshold = thresholds.get(category, 4)
            if severity >= threshold:
                blocked_categories.append(category)
        
        is_safe = len(blocked_categories) == 0
        
        return {
            "safe": is_safe,
            "categories": categories,
            "severity": max_severity,
            "thresholds": thresholds,
            "blocked_categories": blocked_categories
        }

class AzureLanguageService:
    """Azure Language Service for PII detection and masking with Service Principal auth"""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        self.tenant_id = os.getenv("TENANT_ID")
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.pii_enabled = os.getenv("PII_MASKING_ENABLED", "true").lower() == "true"
        
        self.enabled = bool(self.endpoint and self.tenant_id and self.client_id and self.client_secret and self.pii_enabled)
        self.credential = None
        
        if self.enabled:
            self.credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
        
        # PII categories to detect
        self.pii_categories = [
            "Person", "PersonType", "PhoneNumber", "Email", "URL", 
            "IPAddress", "DateTime", "Quantity", "Organization",
            "Address", "CreditCardNumber", "ABARoutingNumber",
            "USSocialSecurityNumber", "InternationalBankingAccountNumber"
        ]
        
    def is_enabled(self) -> bool:
        """Check if Language Service is enabled"""
        return self.enabled
    
    async def _get_access_token(self) -> str:
        """Get access token using Service Principal"""
        if not self.enabled:
            raise Exception("Azure Language Service not properly configured")
        
        try:
            # Azure Cognitive Services scope
            scope = "https://cognitiveservices.azure.com/.default"
            token = await self.credential.get_token(scope)
            return token.token
        except Exception as e:
            print(f"Error getting access token for Language Service: {e}")
            raise
    
    async def detect_and_mask_pii(self, text: str) -> Dict[str, Any]:
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
            pii_result = await self._detect_pii(text)
            
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
    
    async def _detect_pii(self, text: str) -> Dict[str, Any]:
        """Detect PII entities in text"""
        access_token = await self._get_access_token()
        url = f"{self.endpoint}/language/:analyze-text?api-version=2022-05-01"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
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
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=10) as response:
                if response.status == 200:
                    result = await response.json()
                    documents = result.get("results", {}).get("documents", [])
                    if documents:
                        return {
                            "entities": documents[0].get("entities", []),
                            "redacted_text": documents[0].get("redactedText", text)
                        }
                else:
                    error_text = await response.text()
                    print(f"PII Detection API error: {response.status} - {error_text}")
        
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
            "ABARoutingNumber": "[ROUTING_NUMBER]",
            "USSocialSecurityNumber": "[SSN]",
            "InternationalBankingAccountNumber": "[IBAN]"
        }
        
        return masks.get(category, "[REDACTED]")

class SafetyManager:
    """Main safety manager combining both services with async support"""
    
    def __init__(self):
        self.content_safety = AzureContentSafety()
        self.language_service = AzureLanguageService()
        
    async def check_content_safety(self, text: str, user_id: int = None) -> Dict[str, Any]:
        """Check content safety and PII in one call"""
        result = {
            "original_content": text,
            "processed_content": text,
            "blocked": False,
            "pii_detected": False,
            "content_safety_result": {},
            "pii_result": {},
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id
        }
        
        try:
            # Step 1: Content Safety Check
            if self.content_safety.is_enabled():
                safety_result = await self.content_safety.analyze_text(text)
                result["content_safety_result"] = safety_result
                
                if not safety_result["safe"]:
                    result["blocked"] = True
                    result["block_reason"] = f"Content blocked due to: {', '.join(safety_result.get('blocked_categories', []))}"
                    return result
            
            # Step 2: PII Detection and Masking (only if content is safe)
            if self.language_service.is_enabled():
                pii_result = await self.language_service.detect_and_mask_pii(text)
                result["pii_result"] = pii_result
                result["pii_detected"] = pii_result["pii_detected"]
                result["processed_content"] = pii_result["masked_text"]
            
            return result
            
        except Exception as e:
            print(f"Safety processing error: {e}")
            result["error"] = str(e)
            # Return safe result on error to avoid blocking chat
            return result
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety service status"""
        return {
            "content_safety_enabled": self.content_safety.is_enabled(),
            "pii_detection_enabled": self.language_service.is_enabled(),
            "content_safety_thresholds": {
                "hate": self.content_safety.hate_threshold if self.content_safety.is_enabled() else None,
                "violence": self.content_safety.violence_threshold if self.content_safety.is_enabled() else None,
                "sexual": self.content_safety.sexual_threshold if self.content_safety.is_enabled() else None,
                "self_harm": self.content_safety.self_harm_threshold if self.content_safety.is_enabled() else None
            },
            "services_status": {
                "content_safety": "enabled" if self.content_safety.is_enabled() else "disabled",
                "language_service": "enabled" if self.language_service.is_enabled() else "disabled"
            }
        }